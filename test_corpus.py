"""Unit tests for ingestion, chunking, and provenance checks."""

import json
from pathlib import Path
import tempfile
import unittest

from sourcegrounded.corpus import Source, chunk_sources, ingest_folder, load_sources, save_sources


class CorpusTests(unittest.TestCase):
    def test_demo_has_eight_proposal_passages(self):
        rows = load_sources("data/demo_sources.json")
        self.assertEqual(len(rows), 8)
        self.assertEqual([s.source_id for s in rows], [f"S{i}" for i in range(1, 9)])
        self.assertIn("Wednesday at 4 PM", rows[0].text)
        self.assertEqual(len(chunk_sources(rows)), 8)

    def test_hash_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "mismatched content hash"):
            Source.from_mapping({
                "source_id": "F1", "title": "Misrepresented source",
                "text": "content", "content_sha256": "fake",
            })

    def test_invalid_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid source_id"):
            Source.from_mapping({"source_id": "../evil", "title": "x", "text": "y"})

    def test_duplicate_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "corpus.json"
            target.write_text(json.dumps({"sources": [
                {"source_id": "X1", "title": "A", "text": "a"},
                {"source_id": "X1", "title": "B", "text": "b"},
            ]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_sources(target)

    def test_ingest_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "documents"
            folder.mkdir()
            (folder / "notes.txt").write_text("The answer is in this local note.", encoding="utf-8")
            manifest = Path(tmp) / "meta.json"
            manifest.write_text(json.dumps({"notes.txt": {
                "source_id": "Notes", "title": "Local Note",
                "url": "https://example.org/example",
            }}), encoding="utf-8")
            sources = ingest_folder(folder, manifest)
            self.assertEqual(sources[0].source_id, "Notes")
            out = Path(tmp) / "saved.json"
            save_sources(out, sources)
            self.assertEqual(load_sources(out), sources)

    def test_chunk_ids_preserve_document_id(self):
        content = " ".join(f"word{i}" for i in range(100))
        item = Source.from_mapping({"source_id": "Doc1", "title": "test", "text": content})
        parts = chunk_sources([item], chunk_words=30, overlap_words=5)
        self.assertEqual(parts[0].passage_id, "Doc1-P1")
        self.assertGreater(len(parts), 2)
        self.assertTrue(all(p.source_id == "Doc1" for p in parts))


if __name__ == "__main__":
    unittest.main()
