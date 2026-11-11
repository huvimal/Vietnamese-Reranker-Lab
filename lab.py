"""Reproducible train / evaluate / search commands."""
import argparse
import json
import sys
import time
from pathlib import Path

from ranking import (BM25, FEATURES, evaluate, fingerprint, grouped_bootstrap_delta,
                     load_collection, mine_pairs, pairwise_fit, rerank)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "collection.json"
MODEL = ROOT / "models" / "reranker.json"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare():
    data = load_collection(DATA)
    return data, BM25(data["documents"])


def check_model(data):
    if not MODEL.exists():
        raise ValueError("Model missing. Run: python lab.py train")
    model = json.loads(MODEL.read_text(encoding="utf-8"))
    if model["collection_sha256"] != fingerprint(data) or model["features"] != FEATURES:
        raise ValueError("Collection or feature schema changed. Retrain before evaluating.")
    if model["implementation_sha256"] != fingerprint((ROOT / "ranking.py").read_text(encoding="utf-8")):
        raise ValueError("Ranking implementation changed. Retrain before evaluating.")
    return model


def train(args):
    data, index = prepare()
    train_queries = [q for q in data["queries"] if q["split"] == "train"]
    dev_queries = [q for q in data["queries"] if q["split"] == "dev"]
    started = time.perf_counter()
    pairs, missed = mine_pairs(index, train_queries, args.candidates, args.negatives)
    experiments, selected = [], {}
    for feature_set in ["full", "lexical"]:
        current = pairs if feature_set == "full" else [
            (p[:5] + [0.0] * (len(p)-5), n[:5] + [0.0] * (len(n)-5)) for p, n in pairs]
        best = None
        for l2 in [.001, .01, .1]:
            weights, history = pairwise_fit(current, epochs=args.epochs, l2=l2)
            summary, _ = evaluate(index, dev_queries, weights, args.candidates, feature_set == "lexical")
            record = {"feature_set": feature_set, "l2": l2, "dev": summary,
                      "initial_epoch_loss": history[0], "final_epoch_loss": history[-1]}
            experiments.append(record)
            if best is None or summary["ndcg_at_5"] > best["dev"]["ndcg_at_5"]:
                best = {**record, "weights": weights, "loss_history": history}
        selected[feature_set] = best
    model = {"schema": 1, "model_type": "pairwise_linear", "features": FEATURES,
             "collection_sha256": fingerprint(data),
             "implementation_sha256": fingerprint((ROOT / "ranking.py").read_text(encoding="utf-8")),
             "training_queries_sha256": fingerprint(train_queries),
             "candidates": args.candidates, "negatives_per_positive": args.negatives,
             "epochs": args.epochs, "seed": 42, "train_queries": len(train_queries),
             "dev_queries": len(dev_queries), "pairs": len(pairs),
             "train_queries_without_retrieved_positive": missed,
             "selection": "highest dev nDCG@5; first l2 wins ties; test not used",
             "models": selected}
    save(MODEL, model)
    save(ROOT / "reports" / "training.json", {"experiments": experiments,
         "elapsed_seconds": time.perf_counter() - started,
         "model_sha256": fingerprint(model), "pairs": len(pairs)})
    print(f"Train: {len(train_queries)} | Dev: {len(dev_queries)} | Pairs: {len(pairs)} | Missed positives: {missed}")
    for name, choice in selected.items():
        print(f"{name}: l2={choice['l2']} | dev nDCG@5={choice['dev']['ndcg_at_5']:.4f}")
    print(f"Model: {MODEL}\nTraining report: {ROOT / 'reports' / 'training.json'}")


def run_evaluation(args):
    data, index = prepare()
    model = check_model(data)
    queries = [q for q in data["queries"] if q["split"] == args.split]
    summaries, detailed = {}, {}
    for name in ["bm25", "lexical", "full"]:
        weights = None if name == "bm25" else model["models"][name]["weights"]
        started = time.perf_counter()
        summary, rows = evaluate(index, queries, weights, model["candidates"], name == "lexical")
        summary["mean_query_ms"] = (time.perf_counter() - started) * 1000 / len(queries)
        summaries[name], detailed[name] = summary, rows
    deltas = grouped_bootstrap_delta(detailed["bm25"], detailed["full"])
    failures = [{"query_id": q["query_id"], "query": q["query"],
                 "baseline_ranked_ids": b["ranked_ids"][:5], "reranked_ids": q["ranked_ids"][:5],
                 "baseline_ndcg_at_5": b["ndcg_at_5"], "reranker_ndcg_at_5": q["ndcg_at_5"]}
                for b, q in zip(detailed["bm25"], detailed["full"])
                if q["ndcg_at_5"] < 1 or q["ndcg_at_5"] < b["ndcg_at_5"]]
    report = {"split": args.split, "synthetic": data.get("synthetic", False),
              "queries": len(queries), "product_groups": len({q['group'] for q in queries}),
              "collection_sha256": fingerprint(data), "model_sha256": fingerprint(model),
              "candidates": model["candidates"], "metrics": summaries,
              "paired_delta": deltas, "failures": failures, "per_query": detailed,
              "limitations": ["Synthetic templates, no claims of real-world Vietnamese search quality.",
                              "Product-disjoint splits share wording templates and feature-design assumptions.",
                              "Only a few test groups; confidence interval is illustrative.",
                              "Candidate recall limits every reranker; no gold insertion during evaluation.",
                              "Latency is a local single-process measurement, not a load test."]}
    save(ROOT / "reports" / f"{args.split}.json", report)
    print(f"Split: {args.split} | Queries: {len(queries)} | Candidate count: {model['candidates']}")
    for name, summary in summaries.items():
        print(f"{name:8} nDCG@5={summary['ndcg_at_5']:.4f}  MRR@5={summary['mrr_at_5']:.4f}  "
              f"Recall@5={summary['recall_at_5']:.4f}  CandidateRecall={summary['candidate_recall']:.4f}")
    print(f"Full - BM25 nDCG delta: {deltas['mean_delta']:.4f}; group bootstrap 95% CI: {deltas['ci95']}")
    print(f"Report: {ROOT / 'reports' / (args.split + '.json')}")


def search(args):
    if not args.query.strip():
        raise ValueError("Query cannot be empty.")
    data, index = prepare()
    model = check_model(data)
    initial = index.retrieve(args.query, model["candidates"])
    ranked = rerank(index, args.query, initial, model["models"]["full"]["weights"])
    for label, result in [("BM25", initial), ("RERANKED", ranked)]:
        print(f"\n{label}")
        for i, (did, score) in enumerate(result[:5], 1):
            print(f"{i}. {did} | score={score:.4f} | {index.docs[did]['text']}")
    print("\nScores are ranking scores, not probabilities. No answer generation is performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--candidates", type=int, default=20)
    p.add_argument("--negatives", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.set_defaults(func=train)
    p = sub.add_parser("evaluate")
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.set_defaults(func=run_evaluation)
    p = sub.add_parser("search")
    p.add_argument("--query", required=True)
    p.set_defaults(func=search)
    args = parser.parse_args()
    if args.command == "train" and (args.candidates < 2 or args.negatives < 1 or args.epochs < 1):
        parser.error("candidates >= 2, negatives >= 1, epochs >= 1 required")
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
