#!/usr/bin/env python3
"""Convert Alpaca JSONL to ShareGPT conversations JSONL.

This converter is intentionally dependency-free so it can run on a fresh VM
before installing the full MedicalGPT training stack.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Alpaca JSONL to ShareGPT conversations JSONL."
    )
    parser.add_argument("--input", required=True, help="Input Alpaca JSONL file.")
    parser.add_argument("--output", required=True, help="Output ShareGPT JSONL file.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10000,
        help="Print progress every N input rows.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_user_prompt(instruction: str, input_text: str) -> str:
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def convert_record(record: dict[str, Any]) -> dict[str, Any] | None:
    instruction = clean_text(record.get("instruction"))
    input_text = clean_text(record.get("input"))
    output = clean_text(record.get("output"))
    user_prompt = build_user_prompt(instruction, input_text)

    if not user_prompt or not output:
        return None

    return {
        "conversations": [
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": output},
        ]
    }


def convert_file(input_path: Path, output_path: Path, log_every: int) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    total = 0
    kept = 0
    skipped = 0
    bad_json = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                skipped += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                skipped += 1
                continue

            converted = convert_record(record)
            if converted is None:
                skipped += 1
                continue

            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            kept += 1

            if log_every > 0 and total % log_every == 0:
                elapsed = time.time() - started_at
                speed = total / elapsed if elapsed > 0 else 0.0
                print(
                    f"[progress] read={total} kept={kept} skipped={skipped} "
                    f"speed={speed:.1f} rows/s",
                    flush=True,
                )

    elapsed = time.time() - started_at
    return {
        "input": str(input_path),
        "output": str(output_path),
        "total_read": total,
        "kept": kept,
        "skipped": skipped,
        "bad_json": bad_json,
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> None:
    args = parse_args()
    report = convert_file(Path(args.input), Path(args.output), args.log_every)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
