"""Command-line entry point: python -m sourcegrounded.cli --help."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .corpus import chunk_sources, ingest_folder, load_sources, save_sources
from .evaluation import (
    evaluate_retrieval, load_cases, run_generation_evaluation, score_human_annotations,
)
from .llm import ChatClient, ProviderError, load_local_env
from .pipeline import MODES, TextGenerationPipeline
from .retrieval import SemanticRetriever, TfidfRetriever


def _pipeline(args: argparse.Namespace) -> TextGenerationPipeline:
    sources = load_sources(args.sources)
    chunks = chunk_sources(sources)
    if args.retriever == "tfidf":
        index = TfidfRetriever(chunks)
    else:
        index = SemanticRetriever(chunks)
    return TextGenerationPipeline(
        sources=sources, model_client=ChatClient(),
        retriever=index, top_k=args.top_k, min_score=args.min_score,
    )


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    parser = argparse.ArgumentParser(
        description="Source-grounded generative AI learning assistant (AAI 6665)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--sources", default="data/demo_sources.json", help="Local JSON corpus")
        p.add_argument("--retriever", choices=["tfidf", "semantic"], default="tfidf")
        p.add_argument("--top-k", type=int, default=3)
        p.add_argument("--min-score", type=float, default=0.08)

    retrieve = sub.add_parser("retrieve", help="Offline retrieval; no API key or requests")
    retrieve.add_argument("question")
    common(retrieve)

    ask = sub.add_parser("ask", help="Generate an answer (requires an API key)")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=MODES, default="guarded-rag")
    common(ask)

    lookup = sub.add_parser("evaluate-retrieval", help="Offline recall@3 diagnostic")
    lookup.add_argument("--cases", default="data/eval_36.json")
    common(lookup)

    evaluate = sub.add_parser("evaluate", help="Real generation calls, export review materials")
    evaluate.add_argument("--cases", default="data/eval_36.json")
    evaluate.add_argument("--out", default="outputs/review_run")
    evaluate.add_argument("--limit", type=int, default=3)
    evaluate.add_argument("--repeats", type=int, default=1)
    evaluate.add_argument(
        "--confirm-api-calls", action="store_true",
        help="Required to avoid accidental provider cost / data transmission",
    )
    common(evaluate)

    review = sub.add_parser("score-human", help="Aggregate manually reviewed CSV labels")
    review.add_argument("--csv", required=True)

    ingest = sub.add_parser("ingest", help="Build a vetted corpus from local files")
    ingest.add_argument("--directory", required=True)
    ingest.add_argument("--manifest", help="Optional provenance JSON keyed by filename")
    ingest.add_argument("--out", default="data/my_sources.json")

    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            sources = ingest_folder(args.directory, args.manifest)
            save_sources(args.out, sources)
            _json({"sources_added": len(sources), "path": str(Path(args.out).resolve())})
        elif args.command == "score-human":
            _json(score_human_annotations(args.csv))
        elif args.command == "retrieve":
            _json([hit.to_mapping() for hit in _pipeline(args).retrieve(args.question)])
        elif args.command == "ask":
            _json(_pipeline(args).answer(args.question, mode=args.mode).to_mapping())
        elif args.command == "evaluate-retrieval":
            _json(evaluate_retrieval(_pipeline(args), load_cases(args.cases)))
        elif args.command == "evaluate":
            if not args.confirm_api_calls:
                parser.error("Generation calls require --confirm-api-calls.")
            if args.limit > 36:
                parser.error("--limit must be <= 36")
            _json(run_generation_evaluation(
                _pipeline(args), load_cases(args.cases), args.out, args.repeats, args.limit
            ))
        return 0
    except (ValueError, OSError, RuntimeError, ProviderError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
