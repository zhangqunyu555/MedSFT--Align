#!/usr/bin/env python3
"""Build a fixed 1K long-text medical evaluation set for PPL.

The output is still Alpaca-style JSONL, but each row is enriched with stable
metadata so the same set can be reused across base / SFT / PPO models.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed medical long-text PPL eval set.")
    parser.add_argument("--input", required=True, help="Input Alpaca JSONL file.")
    parser.add_argument("--output", required=True, help="Output PPL eval JSONL file.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of samples to keep.")
    parser.add_argument(
        "--min-output-chars",
        type=int,
        default=200,
        help="Drop answers shorter than this before ranking by length.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50000,
        help="Print progress every N rows.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def iter_alpaca_rows(path: Path, log_every: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_at = time.time()
    rows: list[dict[str, Any]] = []
    total = 0
    bad_json = 0
    missing_fields = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            instruction = clean_text(row.get("instruction"))
            input_text = clean_text(row.get("input"))
            output = clean_text(row.get("output"))
            if not (instruction or input_text) or not output:
                missing_fields += 1
                continue

            rows.append(
                {
                    "id": row.get("id") or f"ppl-candidate-{line_no}",
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                    "source": row.get("source", "cleaned_alpaca"),
                    "source_row_id": row.get("row_id", row.get("source_row_id", line_no)),
                    "output_char_length": len(output),
                    "input_char_length": len(instruction) + len(input_text),
                }
            )

            if log_every > 0 and total % log_every == 0:
                elapsed = time.time() - started_at
                speed = total / elapsed if elapsed > 0 else 0.0
                print(
                    f"[progress] read={total} valid={len(rows)} bad_json={bad_json} "
                    f"missing={missing_fields} speed={speed:.1f} rows/s",
                    flush=True,
                )

    report = {
        "input": str(path),
        "total_read": total,
        "valid_candidates": len(rows),
        "bad_json": bad_json,
        "missing_fields": missing_fields,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    return rows, report


def build_eval_set(input_path: Path, output_path: Path, sample_size: int, min_output_chars: int, log_every: int) -> dict[str, Any]:
    rows, report = iter_alpaca_rows(input_path, log_every)
    rows = [row for row in rows if row["output_char_length"] >= min_output_chars]
    rows.sort(key=lambda row: (-row["output_char_length"], str(row["id"])))
    selected = rows[:sample_size]

    if len(selected) != sample_size:
        raise SystemExit(
            f"Not enough long-text samples: need={sample_size}, got={len(selected)}, "
            f"min_output_chars={min_output_chars}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for new_id, row in enumerate(selected):
            row = dict(row)
            row["id"] = f"medical-longtext-ppl-{new_id}"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = [row["output_char_length"] for row in selected]
    report.update(
        {
            "output": str(output_path),
            "sample_size": sample_size,
            "min_output_chars": min_output_chars,
            "selected_count": len(selected),
            "selected_min_output_chars": min(lengths),
            "selected_max_output_chars": max(lengths),
            "selected_avg_output_chars": round(sum(lengths) / len(lengths), 2),
            "selection_strategy": "sort_by_output_char_length_desc_then_id",
        }
    )
    return report


def main() -> None:
    args = parse_args()
    report = build_eval_set(
        input_path=Path(args.input),
        output_path=Path(args.output),
        sample_size=args.sample_size,
        min_output_chars=args.min_output_chars,
        log_every=args.log_every,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
