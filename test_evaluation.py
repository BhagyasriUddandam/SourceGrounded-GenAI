import csv
from pathlib import Path
import tempfile
import unittest

from sourcegrounded.corpus import load_sources
from sourcegrounded.evaluation import (
    evaluate_retrieval, load_cases, score_human_annotations,
)
from sourcegrounded.pipeline import TextGenerationPipeline


class EvaluationTests(unittest.TestCase):
    def test_benchmark_balanced(self):
        cases = load_cases("data/eval_36.json")
        self.assertEqual(len(cases), 36)
        for category in ("answerable", "partial", "unanswerable"):
            self.assertEqual(sum(c["category"] == category for c in cases), 12)

    def test_retrieval_metrics_do_not_need_api(self):
        pipeline = TextGenerationPipeline(load_sources("data/demo_sources.json"), model_client=None)
        result = evaluate_retrieval(pipeline, load_cases("data/eval_36.json"))
        self.assertEqual(result["gold_evaluable_count"], 24)
        self.assertTrue(0 <= result["query_recall_at_k"] <= 1)

    def test_human_review_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            with path.open("w", newline="", encoding="utf-8") as out:
                writer = csv.DictWriter(out, fieldnames=[
                    "mode", "supported_claims", "unsupported_claims",
                    "valid_citations", "total_citations",
                    "usefulness_1_to_5", "abstention_appropriate_0_or_1",
                ])
                writer.writeheader()
                writer.writerow({"mode": "guarded-rag"})
            result = score_human_annotations(path)
            self.assertIsNone(result["modes"]["guarded-rag"]["citation_precision"])
            self.assertIsNone(result["modes"]["guarded-rag"]["supported_claim_rate"])


if __name__ == "__main__":
    unittest.main()
