import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from make_data import build
from ranking import (BM25, FEATURES, evaluate, features, grouped_bootstrap_delta,
                     load_collection, metrics, mine_pairs, pairwise_fit, rerank, tokens)


class RankingTests(unittest.TestCase):
    def test_vietnamese_unicode(self):
        self.assertEqual(tokens("ĐỔI trả"), ["doi", "tra"])
        self.assertEqual(tokens("ĐỔI trả"), tokens("ĐỔI trả"))

    def test_group_leakage_rejected(self):
        data = build()
        data["queries"][1]["split"] = "dev" if data["queries"][0]["split"] != "dev" else "test"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "collection.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_collection(path)

    def test_dataset_valid_and_reproducible(self):
        data = build()
        self.assertEqual(data, build())
        self.assertEqual(len(data["documents"]), 180)
        self.assertEqual(len(data["queries"]), 360)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "collection.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(load_collection(path), data)

    def test_metrics_known_ranks(self):
        qrels = {"right": 1}
        self.assertEqual(metrics(["right"], qrels)["ndcg_at_5"], 1)
        row = metrics(["wrong", "right"], qrels)
        self.assertEqual(row["mrr_at_5"], .5)
        self.assertAlmostEqual(row["ndcg_at_5"], 1 / math.log2(3))
        self.assertEqual(metrics(["wrong"], qrels)["recall_at_5"], 0)

    def test_bm25_tie_order_deterministic(self):
        index = BM25([{"id": "b", "title": "", "text": "hello"},
                      {"id": "a", "title": "", "text": "hello"}])
        self.assertEqual([d for d, _ in index.retrieve("missing")], ["a", "b"])

    def test_pairwise_learning_and_determinism(self):
        p, n = [1.0] + [0.0] * (len(FEATURES)-1), [0.0] * len(FEATURES)
        w, history = pairwise_fit([(p, n)], epochs=30)
        self.assertGreater(w[0], 0)
        self.assertLess(history[-1], history[0])
        self.assertEqual((w, history), pairwise_fit([(p, n)], epochs=30))

    def test_no_qrels_or_id_in_features(self):
        doc = {"id": "a", "title": "Bảo hành SP001", "text": "Sửa miễn phí"}
        changed = {**doc, "id": "different", "group": "other", "qrels": {"x": 100}}
        self.assertEqual(features("Sửa SP001", doc, 1), features("Sửa SP001", changed, 1))
        self.assertEqual(len(features("", doc, 0)), len(FEATURES))

    def test_train_miner_rejects_dev_and_test(self):
        data = build()
        index = BM25(data["documents"])
        for split in ["dev", "test"]:
            query = next(q for q in data["queries"] if q["split"] == split)
            with self.assertRaisesRegex(ValueError, "train"):
                mine_pairs(index, [query])

    def test_reranker_cannot_invent_candidates(self):
        data = build()
        index = BM25(data["documents"])
        initial = index.retrieve("Hoàn tiền SP001", 3)
        result = rerank(index, "Hoàn tiền SP001", initial, [1.0] * len(FEATURES))
        self.assertEqual({d for d, _ in initial}, {d for d, _ in result})

    def test_missing_positive_not_injected(self):
        data = build()
        index = BM25(data["documents"])
        query = copy.deepcopy(data["queries"][0])
        first = index.retrieve(query["text"], 1)[0][0]
        other = next(d["id"] for d in data["documents"] if d["id"] != first)
        query["qrels"] = {other: 1}
        summary, rows = evaluate(index, [query], [1.0]*len(FEATURES), candidates=1)
        self.assertEqual(summary["candidate_recall"], 0)
        self.assertNotIn(other, rows[0]["ranked_ids"])

    def test_bootstrap_groups_and_alignment(self):
        base = [{"query_id": "q1", "group": "a", "ndcg_at_5": 0},
                {"query_id": "q2", "group": "b", "ndcg_at_5": 0}]
        better = [{**r, "ndcg_at_5": 1} for r in base]
        result = grouped_bootstrap_delta(base, better, samples=100)
        self.assertEqual(result["ci95"], [1, 1])
        self.assertEqual(result["groups"], 2)
        with self.assertRaises(ValueError):
            grouped_bootstrap_delta(base, better[::-1])


if __name__ == "__main__":
    unittest.main()
