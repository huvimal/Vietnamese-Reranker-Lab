"""BM25 and a trainable pairwise linear reranker. Standard library only."""
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path


def normalize(text):
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def tokens(text):
    return re.findall(r"[a-z0-9]+", normalize(text))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_collection(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    docs, queries = data["documents"], data["queries"]
    ids = {d["id"] for d in docs}
    if not docs or len(ids) != len(docs):
        raise ValueError("Documents must be nonempty and have unique IDs.")
    if len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Query IDs must be unique.")
    group_splits = {}
    doc_groups = {d["id"]: d["group"] for d in docs}
    for q in queries:
        if q["split"] not in {"train", "dev", "test"}:
            raise ValueError("Unknown split.")
        if q["group"] in group_splits and group_splits[q["group"]] != q["split"]:
            raise ValueError("Group leakage across splits.")
        group_splits[q["group"]] = q["split"]
        if not q["qrels"] or not any(v > 0 for v in q["qrels"].values()):
            raise ValueError("Every query needs a positive relevance label.")
        for did, grade in q["qrels"].items():
            if did not in ids or not isinstance(grade, int) or isinstance(grade, bool) or grade < 0:
                raise ValueError("Invalid relevance label or document ID.")
            if grade > 0 and doc_groups[did] != q["group"]:
                raise ValueError("Positive document must belong to the query group.")
    if {q["split"] for q in queries} != {"train", "dev", "test"}:
        raise ValueError("All three splits are required.")
    return data


class BM25:
    def __init__(self, documents):
        self.docs = {d["id"]: d for d in documents}
        self.counts = {did: Counter(tokens(d["title"] + " " + d["text"]))
                       for did, d in self.docs.items()}
        self.lengths = {did: sum(c.values()) for did, c in self.counts.items()}
        self.average = sum(self.lengths.values()) / max(1, len(self.docs))
        df = Counter(t for c in self.counts.values() for t in c)
        self.idf = {t: math.log(1 + (len(self.docs) - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}

    def score(self, query, did):
        counts = self.counts[did]
        norm = 1.2 * (0.25 + 0.75 * self.lengths[did] / max(1, self.average))
        return sum(self.idf.get(t, 0) * counts[t] * 2.2 / (counts[t] + norm)
                   for t in set(tokens(query)) if counts[t])

    def retrieve(self, query, count=20):
        if count < 1:
            raise ValueError("Candidate count must be positive.")
        return sorted([(did, self.score(query, did)) for did in self.docs],
                      key=lambda x: (-x[1], x[0]))[:count]


# These cues are manually engineered text features, never gold labels or document IDs.
# The same feature extractor is used for training, evaluation, and free-text search.
INTENT_CUES = {
    "return": ["doi tra", "gui lai", "tra lai", "nguyen tem"],
    "warranty": ["bao hanh", "sua", "loi nha san xuat", "nha san xuat"],
    "shipping": ["giao hang", "van chuyen", "nhan duoc hang", "mien cuoc", "mien phi giao"],
    "refund": ["hoan tien", "tien ve", "chuyen lai", "sau khi duyet"],
    "cancel": ["huy don", "dung don", "ngung don"],
    "payment": ["thanh toan", "phuong thuc", "tien mat", "chuyen khoan", "the"],
}
FEATURES = ["bm25", "query_coverage", "title_coverage", "bigram_overlap",
            "char_similarity", "product_match", "product_conflict", "length_ratio"] + list(INTENT_CUES)


def cues(text, phrases):
    return any(f" {p} " in f" {text} " for p in phrases)


def features(query, document, bm25):
    q, d = tokens(query), tokens(document["title"] + " " + document["text"])
    qs, ds = set(q), set(d)
    title = set(tokens(document["title"]))
    qb, db = set(zip(q, q[1:])), set(zip(d, d[1:]))
    qn, dn = " ".join(q), " ".join(d)
    qc = {qn[i:i+3] for i in range(len(qn)-2)}
    dc = {dn[i:i+3] for i in range(len(dn)-2)}
    qp, dp = {t for t in qs if re.fullmatch(r"sp\d+", t)}, {t for t in ds if re.fullmatch(r"sp\d+", t)}
    values = [math.log1p(max(0, bm25)), len(qs & ds) / max(1, len(qs)),
              len(qs & title) / max(1, len(qs)), len(qb & db) / max(1, len(qb)),
              len(qc & dc) / max(1, len(qc | dc)), float(bool(qp & dp)),
              float(bool(qp) and not bool(qp & dp)), min(len(q), len(d)) / max(1, len(q), len(d))]
    values += [float(cues(qn, phrases) and cues(dn, phrases)) for phrases in INTENT_CUES.values()]
    return values


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def pairwise_fit(pairs, epochs=60, learning_rate=0.05, l2=0.01, seed=42):
    if not pairs:
        raise ValueError("No training pairs. Check candidate recall and labels.")
    weights = [0.0] * len(FEATURES)
    rng = random.Random(seed)
    loss_history = []
    for epoch in range(epochs):
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        rate = learning_rate / math.sqrt(1 + epoch / 10)
        for positive, negative in shuffled:
            delta = [p - n for p, n in zip(positive, negative)]
            margin = max(-40, min(40, dot(weights, delta)))
            gradient_scale = -1 / (1 + math.exp(margin))
            weights = [w - rate * (gradient_scale * x + l2 * w) for w, x in zip(weights, delta)]
        losses = []
        for p, n in pairs:
            z = -dot(weights, [x - y for x, y in zip(p, n)])
            losses.append(max(0, z) + math.log1p(math.exp(-abs(z))))
        loss_history.append(sum(losses) / len(losses))
    return weights, loss_history


def mine_pairs(index, queries, candidates=20, negatives=5):
    pairs, missed = [], 0
    for q in queries:
        if q["split"] != "train":
            raise ValueError("Hard-negative mining only accepts train queries.")
        retrieved = index.retrieve(q["text"], candidates)
        positive = [(d, s) for d, s in retrieved if q["qrels"].get(d, 0) > 0]
        negative = [(d, s) for d, s in retrieved if q["qrels"].get(d, 0) == 0][:negatives]
        if not positive:
            missed += 1
        for pd, ps in positive:
            for nd, ns in negative:
                pairs.append((features(q["text"], index.docs[pd], ps),
                              features(q["text"], index.docs[nd], ns)))
    return pairs, missed


def rerank(index, query, candidates, weights, lexical_only=False):
    scores = []
    for did, score in candidates:
        x = features(query, index.docs[did], score)
        if lexical_only:
            x[5:] = [0.0] * (len(x) - 5)
        scores.append((did, dot(weights, x)))
    return sorted(scores, key=lambda x: (-x[1], x[0]))


def metrics(ranked_ids, qrels, k=5):
    positives = {d for d, grade in qrels.items() if grade > 0}
    ideal = sorted(qrels.values(), reverse=True)[:k]
    dcg = sum((2 ** qrels.get(d, 0) - 1) / math.log2(i + 2)
              for i, d in enumerate(ranked_ids[:k]))
    idcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
    reciprocal = next((1 / (i + 1) for i, d in enumerate(ranked_ids[:k]) if d in positives), 0)
    return {"ndcg_at_5": dcg / idcg if idcg else 0,
            "mrr_at_5": reciprocal,
            "recall_at_5": len(set(ranked_ids[:k]) & positives) / max(1, len(positives))}


def mean_metrics(rows):
    keys = ["ndcg_at_5", "mrr_at_5", "recall_at_5", "candidate_recall"]
    return {key: sum(r[key] for r in rows) / max(1, len(rows)) for key in keys}


def evaluate(index, queries, weights=None, candidates=20, lexical_only=False):
    rows = []
    for q in queries:
        initial = index.retrieve(q["text"], candidates)
        ranked = rerank(index, q["text"], initial, weights, lexical_only) if weights is not None else initial
        ids = [d for d, _ in ranked]
        positives = {d for d, grade in q["qrels"].items() if grade > 0}
        row = {"query_id": q["id"], "group": q["group"], "query": q["text"],
               "ranked_ids": ids, **metrics(ids, q["qrels"]),
               "candidate_recall": len(set(ids) & positives) / len(positives)}
        rows.append(row)
    return mean_metrics(rows), rows


def grouped_bootstrap_delta(baseline, candidate, samples=1000, seed=42):
    """Paired group bootstrap, since several queries share the same product."""
    if [r["query_id"] for r in baseline] != [r["query_id"] for r in candidate]:
        raise ValueError("Bootstrap rows must contain matching queries in the same order.")
    grouped = {}
    for a, b in zip(baseline, candidate):
        grouped.setdefault(a["group"], []).append(b["ndcg_at_5"] - a["ndcg_at_5"])
    groups = sorted(grouped)
    if not groups:
        raise ValueError("No rows to bootstrap.")
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        values = [v for g in rng.choices(groups, k=len(groups)) for v in grouped[g]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return {"metric": "ndcg_at_5", "method": "paired product-group bootstrap",
            "groups": len(groups), "samples": samples,
            "mean_delta": sum(b["ndcg_at_5"] - a["ndcg_at_5"] for a, b in zip(baseline, candidate)) / len(baseline),
            "ci95": [draws[int(samples * .025)], draws[min(samples-1, int(samples * .975))]]}
