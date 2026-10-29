"""Local document ingestion and provenance-preserving text chunking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import json
import re


@dataclass(frozen=True)
class Source:
    source_id: str
    title: str
    text: str
    url: str = ""
    author: str = ""
    year: str = ""
    source_type: str = "local_document"
    license_note: str = "Check source permissions before redistribution"
    retrieved_at: str = ""
    content_sha256: str = ""

    @classmethod
    def from_mapping(cls, item: dict) -> "Source":
        source_id = str(item.get("source_id", "")).strip()
        title = str(item.get("title", "")).strip()
        content = str(item.get("text", "")).strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,49}", source_id):
            raise ValueError(f"Invalid source_id: {source_id!r}")
        if not title or not content:
            raise ValueError(f"Source {source_id} needs both title and text")
        if len(content) > 2_000_000:
            raise ValueError(f"Source {source_id} exceeds the size limit")
        expected = sha256(content.encode("utf-8")).hexdigest()
        given = str(item.get("content_sha256", "")).strip()
        if given and given != expected:
            raise ValueError(f"Source {source_id} has a mismatched content hash")
        return cls(
            source_id=source_id, title=title, text=content,
            url=str(item.get("url", "")), author=str(item.get("author", "")),
            year=str(item.get("year", "")),
            source_type=str(item.get("source_type", "local_document")),
            license_note=str(item.get("license_note", "Check source permissions before redistribution")),
            retrieved_at=str(item.get("retrieved_at", "")),
            content_sha256=expected,
        )

    def to_mapping(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Passage:
    passage_id: str
    source_id: str
    title: str
    text: str
    url: str
    chunk_index: int
    content_sha256: str

    def to_mapping(self) -> dict:
        return asdict(self)


def load_sources(path: str | Path) -> list[Source]:
    """Read a fully local, vetted JSON collection; no URL fetching occurs."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw["sources"] if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise ValueError("Corpus must contain a nonempty 'sources' list")
    sources = [Source.from_mapping(row) for row in rows]
    ids = [s.source_id for s in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source IDs found")
    return sources


def save_sources(path: str | Path, sources: list[Source]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"sources": [s.to_mapping() for s in sources]}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def chunk_sources(sources: list[Source], chunk_words: int = 450, overlap_words: int = 70) -> list[Passage]:
    """Chunk by approximate words; preserve source identity and content digest."""
    if chunk_words < 20 or not 0 <= overlap_words < chunk_words:
        raise ValueError("chunk_words must be >=20 and overlap must be smaller")
    passages: list[Passage] = []
    for src in sources:
        tokens = src.text.split()
        step = chunk_words - overlap_words
        parts = [tokens[i:i + chunk_words] for i in range(0, len(tokens), step)]
        if len(parts) > 1 and len(parts[-1]) <= overlap_words:
            parts.pop()
        for i, words in enumerate(parts, start=1):
            passages.append(Passage(
                passage_id=src.source_id if len(parts) == 1 else f"{src.source_id}-P{i}",
                source_id=src.source_id, title=src.title, text=" ".join(words),
                url=src.url, chunk_index=i, content_sha256=src.content_sha256,
            ))
    return passages


class _ReadableHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def _extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        value = path.read_text(encoding="utf-8")
    elif ext in {".html", ".htm"}:
        reader = _ReadableHTML()
        reader.feed(path.read_text(encoding="utf-8"))
        value = " ".join(reader.parts)
    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires: pip install -e '.[pdf]'") from exc
        value = "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return re.sub(r"\s+", " ", value).strip()


def ingest_folder(folder: str | Path, manifest_path: str | Path | None = None) -> list[Source]:
    """Import local files only. A manifest can explicitly provide provenance."""
    base = Path(folder)
    if not base.is_dir():
        raise ValueError(f"Not a directory: {base}")
    metadata: dict = {}
    if manifest_path:
        metadata = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    accepted = {".txt", ".md", ".pdf", ".html", ".htm"}
    sources: list[Source] = []
    for file in sorted(base.iterdir()):
        if not file.is_file() or file.suffix.lower() not in accepted:
            continue
        text = _extract_text(file)
        if not text:
            continue
        meta = dict(metadata.get(file.name, {}))
        proposed_id = re.sub(r"[^A-Za-z0-9_-]+", "-", file.stem).strip("-")[:50]
        if not proposed_id or not proposed_id[0].isalpha():
            proposed_id = "D" + proposed_id
        values = {
            "source_id": meta.pop("source_id", proposed_id),
            "title": meta.pop("title", file.stem.replace("_", " ")),
            "text": text,
            "retrieved_at": meta.pop("retrieved_at", date.today().isoformat()),
            **meta,
        }
        sources.append(Source.from_mapping(values))
    if not sources:
        raise ValueError("No readable files found (.txt, .md, .html, .pdf)")
    ids = [s.source_id for s in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source IDs; rename files or set IDs in manifest")
    return sources

