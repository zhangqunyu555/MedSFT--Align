#!/usr/bin/env python3
"""Add a four-section format instruction to medical PPO prompts.

The script is idempotent: rows that already have format_prompt_enabled=True are
kept as-is and will not be prefixed again.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


FORMAT_INSTRUCTION = (
    "请严格按照以下四个小标题回答：\n"
    "1. 病情分析\n"
    "2. 处理建议\n"
    "3. 风险提示\n"
    "4. 就医建议\n\n"
    "病例问题："
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add four-section format prompt to PPO JSONL dataset.")
    parser.add_argument("--input", required=True, help="Input PPO JSONL path.")
    parser.add_argument("--output", required=True, help="Output PPO JSONL path. Can be the same as input.")
    parser.add_argument("--backup", required=True, help="Backup path for the original input JSONL.")
    parser.add_argument("--report", required=True, help="Report JSON path to update or create.")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row at {path}:{line_no}")
            yield row


def add_format_prompt(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    row = dict(row)
    if row.get("format_prompt_enabled") is True:
        row.setdefault("format_instruction", FORMAT_INSTRUCTION)
        row.setdefault("original_prompt", clean_text(row.get("prompt")))
        return row, False

    original_prompt = clean_text(row.get("original_prompt") or row.get("prompt"))
    if not original_prompt:
        raise ValueError(f"Missing prompt for row id={row.get('id')}")

    row["original_prompt"] = original_prompt
    row["format_instruction"] = FORMAT_INSTRUCTION
    row["format_prompt_enabled"] = True
    row["prompt"] = FORMAT_INSTRUCTION + original_prompt
    return row, True


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"previous_report_parse_error": True}


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def convert_dataset(input_path: Path, output_path: Path, backup_path: Path, report_path: Path) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(input_path, backup_path)

    total = 0
    changed = 0
    already_enabled = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(output_path.parent)) as tmp:
        tmp_path = Path(tmp.name)
        for row in iter_jsonl(input_path):
            total += 1
            converted, did_change = add_format_prompt(row)
            if did_change:
                changed += 1
            else:
                already_enabled += 1
            tmp.write(json.dumps(converted, ensure_ascii=False) + "\n")

    tmp_path.replace(output_path)

    report = load_report(report_path)
    report.update(
        {
            "format_prompt_enabled": True,
            "format_prompt_instruction": FORMAT_INSTRUCTION,
            "format_prompt_input": str(input_path),
            "format_prompt_output": str(output_path),
            "format_prompt_backup": str(backup_path),
            "format_prompt_total_rows": total,
            "format_prompt_changed_rows": changed,
            "format_prompt_already_enabled_rows": already_enabled,
        }
    )
    write_report(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = convert_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        backup_path=Path(args.backup),
        report_path=Path(args.report),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
