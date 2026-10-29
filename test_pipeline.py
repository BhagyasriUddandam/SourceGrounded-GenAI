"""Controlled model outputs test guardrails without a paid provider account."""

import json
import unittest

from sourcegrounded.corpus import load_sources
from sourcegrounded.llm import LLMResult
from sourcegrounded.pipeline import (
    ConversationState, EVIDENCE_LIMIT, TextGenerationPipeline,
)


class FakeClient:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.count = 0

    def chat(self, messages):
        self.count += 1
        return LLMResult(
            text=next(self.answers), model="offline-fake",
            usage={"total_tokens": 1}, latency_seconds=0.01,
        )


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = load_sources("data/demo_sources.json")

    def build(self, *responses):
        fake = FakeClient(responses)
        return TextGenerationPipeline(self.sources, fake), fake

    def test_supported_answer_preserves_source_id(self):
        message = json.dumps({
            "answer": "The reading group meets Wednesday at 4 PM [S1].",
            "cited_source_ids": ["S1"], "evidence_status": "supported",
        })
        pipeline, _ = self.build(message)
        result = pipeline.answer("When does the graduate AI reading group meet?")
        self.assertEqual(result.evidence_status, "supported")
        self.assertEqual(result.cited_source_ids, ["S1"])

    def test_unknown_source_citation_is_withheld(self):
        message = json.dumps({
            "answer": "It meets on Mars [S999].",
            "cited_source_ids": ["S999"], "evidence_status": "supported",
        })
        pipeline, _ = self.build(message)
        result = pipeline.answer("When does the reading group meet?")
        self.assertEqual(result.answer, EVIDENCE_LIMIT)
        self.assertEqual(result.evidence_status, "needs_review")

    def test_no_citations_is_withheld(self):
        message = json.dumps({
            "answer": "The reading group meets on Wednesday.",
            "cited_source_ids": [], "evidence_status": "supported",
        })
        pipeline, _ = self.build(message)
        result = pipeline.answer("When does the reading group meet?")
        self.assertEqual(result.evidence_status, "needs_review")

    def test_structured_and_inline_citations_must_match(self):
        message = json.dumps({
            "answer": "The reading group meets on Wednesday [S1].",
            "cited_source_ids": [], "evidence_status": "supported",
        })
        pipeline, _ = self.build(message)
        result = pipeline.answer("When does the reading group meet?")
        self.assertEqual(result.evidence_status, "needs_review")

    def test_invalid_json_withheld(self):
        pipeline, _ = self.build("Unstructured answer with no citation.")
        result = pipeline.answer("Where is the reading group?")
        self.assertEqual(result.evidence_status, "needs_review")

    def test_empty_retrieval_skips_provider(self):
        pipeline, fake = self.build()
        result = pipeline.answer("Astrophysical quasar nebula magnetosphere")
        self.assertEqual(result.evidence_status, "insufficient")
        self.assertEqual(fake.count, 0)

    def test_one_clarification_limit(self):
        message = json.dumps({
            "answer": "Which deadline do you mean?", "cited_source_ids": [],
            "evidence_status": "clarify",
        })
        pipeline, fake = self.build(message, message)
        state = ConversationState()
        question = "What is the deadline for the project poster?"
        self.assertEqual(pipeline.answer(question, state=state).evidence_status, "clarify")
        self.assertEqual(state.clarifications_asked, 1)
        self.assertEqual(pipeline.answer(question, state=state).evidence_status, "insufficient")
        self.assertEqual(fake.count, 2)

    def test_model_only_does_not_retrieve(self):
        pipeline, _ = self.build("General AI is a field of study.")
        result = pipeline.answer("What is general AI?", mode="model-only")
        self.assertEqual(result.evidence_status, "unverified")
        self.assertEqual(result.retrieved, [])

    def test_basic_rag_unverified_citation_warning(self):
        pipeline, _ = self.build("On Wednesday [S1]. Also [S999].")
        result = pipeline.answer("When does the reading group meet?", mode="basic-rag")
        self.assertEqual(result.evidence_status, "not_validated")
        self.assertIn("S999", " ".join(result.warnings))


if __name__ == "__main__":
    unittest.main()
