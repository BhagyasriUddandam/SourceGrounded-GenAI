"""Tests for offline TF-IDF quality and filtering."""

import unittest

from sourcegrounded.corpus import chunk_sources, load_sources
from sourcegrounded.retrieval import TfidfRetriever


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = TfidfRetriever(chunk_sources(load_sources("data/demo_sources.json")))

    def test_reading_group_hit(self):
        top = self.index.search("What is the reading group registration deadline?", top_k=3)
        self.assertIn("S1", [hit.passage.source_id for hit in top])

    def test_api_secret_hit(self):
        top = self.index.search("Where should model API secrets be stored?", top_k=3)
        self.assertTrue(top)
        self.assertEqual(top[0].passage.source_id, "S5")

    def test_unrelated_query_is_empty(self):
        top = self.index.search("Astrophysical quasar nebula magnetosphere", top_k=3)
        self.assertEqual(top, [])

    def test_empty_input_is_empty(self):
        self.assertEqual(self.index.search("the and of"), [])

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            self.index.search("posters", top_k=0)


if __name__ == "__main__":
    unittest.main()
