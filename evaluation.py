"""Reproducible retrieval checks and a transparent human-review workflow.

Automated metrics cannot establish whether generated claims are actually true.
This module deliberately leaves claim-level judgments to human reviewers.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

from .llm import ProviderError
from .pipeline import MODES, ConversationState, TextGenerationPipeline


REVIEW_COLUMNS = [
    "run_id", "question_id", "category", "mode", "repeat",
    "answer", "cited_source_ids", "gold_source_ids",
    "supported_claims", "unsupported_claims", "valid_citations",
    "total_citations", "usefulness_1_to_5", "abstention_appropriate_0_or_1",
    "reviewer_notes",
]


def load_cases(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("Benchmark must contain a nonempty cases list")
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("Each benchmark case must be a JSON object")
        ident = item.get("id")
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError("Case IDs must be nonempty and unique strings")
        seen.add(ident)
        if item.get("category") not in {"answerable", "partial", "unanswerable"}:
            raise ValueError(f"Invalid category for {ident}")
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"Missing question for {ident}")
        if not isinstance(item.get("gold_source_ids"), list) or any(
            not isinstance(v, str) for v in item["gold_source_ids"]
        ):
            raise ValueError(f"Missing gold_source_ids list for {ident}")
    return rows


def evaluate_retrieval(pipeline: TextGenerationPipeline, cases: list[dict]) -> dict:
    """Recall@K only over answerable/partial items with gold evidence.

    If multiple source IDs are needed, count whether any required document was
    recovered. This is query-level recall, not passage-level recall.
    """
    per_case = []
    relevant = []
    for case in cases:
        hits = pipeline.retrieve(case["question"])
        found = {hit.passage.source_id for hit in hits}
        gold = set(case["gold_source_ids"])
        matched = bool(found & gold) if gold else None
        if matched is not None:
            relevant.append(int(matched))
        per_case.append({
            "id": case["id"], "category": case["category"],
            "gold_source_ids": sorted(gold),
            "retrieved_source_ids": [hit.passage.source_id for hit in hits],
            "hit_any_gold": matched,
            "retrieved_irrelevant_on_unanswerable": bool(hits) if not gold else None,
        })
    no_answer = [r for r in per_case if r["category"] == "unanswerable"]
    return {
        "case_count": len(cases),
        "top_k": pipeline.top_k,
        "min_score": pipeline.min_score,
        "gold_evaluable_count": len(relevant),
        "query_recall_at_k": round(sum(relevant) / len(relevant), 4) if relevant else None,
        "unanswerable_with_nonempty_retrieval": sum(
            int(r["retrieved_irrelevant_on_unanswerable"]) for r in no_answer
        ),
        "unanswerable_count": len(no_answer),
        "per_case": per_case,
    }


def run_generation_evaluation(
    pipeline: TextGenerationPipeline, cases: list[dict], output_dir: str | Path,
    repeats: int = 1, limit: int = 3,
) -> dict:
    """Evaluate all three modes, checkpoint each result, and resume safely.

    At 36 items x 3 modes x 2 repeats, up to 216 provider requests are
    planned. Completed rows are never repeated when the same output folder
    is used, even if a prior run was interrupted.
    """
    if repeats < 1 or limit < 1:
        raise ValueError("repeats and limit must be positive")
    selected = cases[:limit]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "run_checkpoint.jsonl"
    config_path = output / "run_config.json"
    signature = sha256(
        json.dumps(
            {
                "cases": selected,
                "source_hashes": [s.content_sha256 for s in pipeline.sources],
                "modes": list(MODES),
                "repeats": repeats,
                "model": getattr(pipeline.client, "model", ""),
                "temperature": getattr(pipeline.client, "temperature", None),
                "top_k": pipeline.top_k,
                "min_score": pipeline.min_score,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("signature") != signature:
            raise ValueError(
                "This output folder belongs to a different evaluation configuration. "
                "Choose a new --out directory to avoid mixing experiments."
            )
    else:
        if checkpoint_path.exists():
            raise ValueError(
                "Found an orphan checkpoint without run_config.json. "
                "Use a fresh --out directory rather than merge unknown runs."
            )
        config = {
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "signature": signature,
            "selected_questions": len(selected),
            "repeats": repeats,
            "model": getattr(pipeline.client, "model", ""),
            "temperature": getattr(pipeline.client, "temperature", None),
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    run_id = config["run_id"]
    previous: dict[tuple[str, str, int], dict] = {}
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") != run_id:
                raise ValueError("Checkpoint run_id mismatch; use a new output directory")
            previous[(row["question_id"], row["mode"], int(row["repeat"]))] = row

    generated: list[dict] = []
    review_rows: list[dict] = []
    total = len(selected) * len(MODES) * repeats
    for case in selected:
        for mode in MODES:
            for rep in range(1, repeats + 1):
                key = (case["id"], mode, rep)
                record = previous.get(key)
                recovered = record is not None and record["evidence_status"] != "provider_error"
                if not recovered:
                    try:
                        # Independent evaluation cases reset the clarification budget.
                        reply = pipeline.answer(case["question"], mode, ConversationState())
                        record = {
                            "run_id": run_id, "question_id": case["id"],
                            "category": case["category"], "repeat": rep,
                            "gold_source_ids": case["gold_source_ids"],
                            **reply.to_mapping(),
                        }
                    except ProviderError as exc:
                        # Keep a transparent missing-output record; do not
                        # mislabel an API failure as a model abstention.
                        record = {
                            "run_id": run_id, "question_id": case["id"],
                            "category": case["category"], "repeat": rep,
                            "gold_source_ids": case["gold_source_ids"],
                            "question": case["question"], "mode": mode,
                            "answer": "", "evidence_status": "provider_error",
                            "cited_source_ids": [], "retrieved": [],
                            "warnings": [str(exc)[:350]], "model": "",
                            "usage": {}, "latency_seconds": 0.0,
                        }
                    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
                        checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
                generated.append(record)
                print(
                    f"[{len(generated):>3}/{total}] {case['id']} {mode} #{rep}: "
                    f"{record['evidence_status']}"
                    + (" (resumed)" if recovered else ""),
                    file=sys.stderr, flush=True,
                )
                review_rows.append({
                    "run_id": run_id, "question_id": case["id"],
                    "category": case["category"], "mode": mode, "repeat": rep,
                    "answer": record["answer"],
                    "cited_source_ids": ";".join(record["cited_source_ids"]),
                    "gold_source_ids": ";".join(case["gold_source_ids"]),
                    "supported_claims": "", "unsupported_claims": "",
                    "valid_citations": "", "total_citations": "",
                    "usefulness_1_to_5": "",
                    "abstention_appropriate_0_or_1": "", "reviewer_notes": "",
                })
    (output / "model_outputs.json").write_text(
        json.dumps(generated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output / "human_annotation_template.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(review_rows)
    summary = {
        "run_id": run_id, "questions_selected": len(selected),
        "repeats": repeats, "modes": list(MODES),
        "output_rows": len(generated),
        "provider_requests": sum(int(bool(r["model"])) for r in generated),
        "provider_requests_note": "Number of rows with a model response, not total network attempts or retries.",
        "provider_error_rows": sum(r["evidence_status"] == "provider_error" for r in generated),
        "checkpoint_path": str(checkpoint_path),
        "resume_enabled": True,
        "outputs_needing_review": sum(r["evidence_status"] == "needs_review" for r in generated),
        "evidence_status_counts": dict(Counter(r["evidence_status"] for r in generated)),
        "human_quality_metrics": "Not measured: fill annotation CSV, then run score-human.",
        "temperature": getattr(pipeline.client, "temperature", None),
        "total_reported_tokens": sum(
            int(r.get("usage", {}).get("total_tokens") or 0) for r in generated
        ),
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def score_human_annotations(path: str | Path) -> dict:
    """Score *only* rows with human-entered labels; blanks are excluded."""
    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    by_mode: dict[str, dict] = {}
    for mode in MODES:
        subset = [r for r in rows if r.get("mode") == mode]
        def pairs(a: str, b: str) -> list[tuple[float, float]]:
            values = []
            for row in subset:
                if row.get(a, "").strip() and row.get(b, "").strip():
                    x, y = float(row[a]), float(row[b])
                    if x < 0 or y < 0:
                        raise ValueError("Counts must be nonnegative")
                    values.append((x, y))
            return values
        claims = pairs("supported_claims", "unsupported_claims")
        cites = pairs("valid_citations", "total_citations")
        if any(v > t for v, t in cites):
            raise ValueError("valid_citations cannot exceed total_citations")
        usefulness = [
            float(r["usefulness_1_to_5"]) for r in subset
            if r.get("usefulness_1_to_5", "").strip()
        ]
        if any(not 1 <= x <= 5 for x in usefulness):
            raise ValueError("Usefulness must be between 1 and 5")
        abstentions = [
            int(r["abstention_appropriate_0_or_1"]) for r in subset
            if r.get("abstention_appropriate_0_or_1", "").strip()
        ]
        if any(x not in (0, 1) for x in abstentions):
            raise ValueError("Abstention labels must be 0 or 1")
        total_claims = sum(x + y for x, y in claims)
        total_citations = sum(y for _, y in cites)
        by_mode[mode] = {
            "total_outputs": len(subset),
            "human_labeled_claim_rows": len(claims),
            "supported_claim_rate": round(sum(x for x, _ in claims) / total_claims, 4)
            if total_claims else None,
            "unsupported_claim_rate": round(sum(y for _, y in claims) / total_claims, 4)
            if total_claims else None,
            "citation_precision": round(sum(x for x, _ in cites) / total_citations, 4)
            if total_citations else None,
            "mean_usefulness": round(sum(usefulness) / len(usefulness), 3) if usefulness else None,
            "appropriate_abstention_rate": round(sum(abstentions) / len(abstentions), 4)
            if abstentions else None,
        }
    return {"modes": by_mode, "warning": "Metrics are based on manually entered labels only."}
