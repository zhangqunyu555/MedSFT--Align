#!/usr/bin/env python3
"""Build paired C-Eval medical JSONL target sets with and without answers."""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATASET = "ceval/ceval-exam"
API_BASE = "https://datasets-server.huggingface.co/rows"
SUBJECT_ZH = {
    "clinical_medicine": "临床医学",
    "basic_medicine": "基础医学",
}
DEFAULT_SUBJECTS = ("clinical_medicine", "basic_medicine")
DEFAULT_SPLITS = ("dev", "val", "test")
CHOICES = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download C-Eval clinical/basic medicine into paired JSONL target sets."
    )
    parser.add_argument(
        "--question-only-output",
        default="data/eval/ceval_medical_question_only.jsonl",
        help="Output JSONL path for target texts without answers.",
    )
    parser.add_argument(
        "--question-answer-output",
        default="data/eval/ceval_medical_question_with_answer.jsonl",
        help="Output JSONL path for target texts with correct answers.",
    )
    parser.add_argument(
        "--report",
        default="data/eval/ceval_medical_target_report.json",
        help="Output report JSON path.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=list(DEFAULT_SUBJECTS),
        help="C-Eval subject configs to download.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to download.",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional maximum rows to keep per subject/split for smoke tests.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between pages.")
    return parser.parse_args()


def fetch_rows(
    subject: str,
    split: str,
    offset: int,
    length: int,
    timeout: int,
) -> dict[str, Any]:
    query = urlencode(
        {
            "dataset": DATASET,
            "config": subject,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{API_BASE}?{query}"
    request = Request(url, headers={"User-Agent": "MedSFT-Align/ceval-target-builder"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_rows_with_retries(
    subject: str,
    split: str,
    offset: int,
    length: int,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fetch_rows(subject, split, offset, length, timeout)
        except (HTTPError, URLError, TimeoutError, http.client.RemoteDisconnected) as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Unexpected retry failure: {last_error}")


def iter_dataset_rows(
    subject: str,
    split: str,
    page_size: int,
    timeout: int,
    retries: int,
    limit: int | None,
    sleep_seconds: float,
) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    stats: Counter = Counter()
    offset = 0

    while True:
        length = page_size
        if limit is not None:
            remaining = limit - len(rows)
            if remaining <= 0:
                break
            length = min(length, remaining)

        payload = fetch_rows_with_retries(subject, split, offset, length, timeout, retries)
        page_rows = payload.get("rows", [])
        total = payload.get("num_rows_total")
        stats["pages"] += 1
        stats["api_rows"] += len(page_rows)

        if not page_rows:
            break

        for item in page_rows:
            row = item.get("row", item)
            if isinstance(row, dict):
                rows.append(row)

        offset += len(page_rows)
        if total is not None and offset >= int(total):
            break
        if len(page_rows) < length:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return rows, stats


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def build_question_with_answer(question: str, answer: str, answer_text: str) -> str:
    return f"{question}\n正确答案：{answer}. {answer_text}"


def build_question_only_target_text(subject_zh: str, question: str) -> str:
    return f"科目：{subject_zh}\n题目：{question}"


def build_question_answer_target_text(subject_zh: str, question: str, answer: str, answer_text: str) -> str:
    return f"科目：{subject_zh}\n题目：{question}\n正确答案：{answer}. {answer_text}"


def convert_row(
    row: dict[str, Any], subject: str, split: str, fallback_index: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    question = clean_text(row.get("question"))
    if not question:
        return None, None, "missing_question"

    options = {choice: clean_text(row.get(choice)) for choice in CHOICES}
    if any(not options[choice] for choice in CHOICES):
        return None, None, "missing_option"

    answer = clean_text(row.get("answer")).upper()
    if not answer:
        return None, None, "missing_answer"
    if answer not in options:
        return None, None, "invalid_answer"

    answer_text = options[answer]
    if not answer_text:
        return None, None, "missing_answer_text"

    row_id = row.get("id", fallback_index)
    subject_zh = SUBJECT_ZH.get(subject, subject)
    record_id = f"{subject}-{split}-{row_id}"
    question_with_answer = build_question_with_answer(question, answer, answer_text)
    question_only_target_text = build_question_only_target_text(subject_zh, question)
    question_answer_target_text = build_question_answer_target_text(
        subject_zh, question, answer, answer_text
    )

    base_record = {
        "id": record_id,
        "source": DATASET,
        "subject": subject,
        "subject_zh": subject_zh,
        "split": split,
        "question": question,
        "options": options,
    }
    question_only_record = {
        **base_record,
        "target_text": question_only_target_text,
    }
    question_answer_record = {
        **base_record,
        "answer": answer,
        "answer_text": answer_text,
        "explanation": clean_text(row.get("explanation")),
        "question_with_answer": question_with_answer,
        "target_text": question_answer_target_text,
    }
    return question_only_record, question_answer_record, None


def write_jsonl_line(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_dataset(args: argparse.Namespace) -> Counter:
    question_only_path = Path(args.question_only_output)
    question_answer_path = Path(args.question_answer_output)
    report_path = Path(args.report)
    question_only_path.parent.mkdir(parents=True, exist_ok=True)
    question_answer_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total_stats: Counter = Counter()
    per_split: dict[str, dict[str, int]] = {}
    errors: list[dict[str, str]] = []

    with question_only_path.open("w", encoding="utf-8") as question_only_file, question_answer_path.open(
        "w", encoding="utf-8"
    ) as question_answer_file:
        for subject in args.subjects:
            for split in args.splits:
                key = f"{subject}/{split}"
                split_stats: Counter = Counter()
                try:
                    rows, api_stats = iter_dataset_rows(
                        subject=subject,
                        split=split,
                        page_size=args.page_size,
                        timeout=args.timeout,
                        retries=args.retries,
                        limit=args.limit_per_split,
                        sleep_seconds=args.sleep,
                    )
                except (HTTPError, URLError, TimeoutError, http.client.RemoteDisconnected) as exc:
                    split_stats["download_failed"] += 1
                    errors.append({"subject": subject, "split": split, "error": str(exc)})
                    per_split[key] = dict(split_stats)
                    total_stats.update(split_stats)
                    continue

                split_stats.update(api_stats)
                split_stats["downloaded"] = len(rows)

                for index, row in enumerate(rows):
                    question_only_record, question_answer_record, reason = convert_row(
                        row, subject, split, index
                    )
                    if reason:
                        split_stats[reason] += 1
                        continue
                    write_jsonl_line(question_only_file, question_only_record)
                    write_jsonl_line(question_answer_file, question_answer_record)
                    split_stats["kept"] += 1

                per_split[key] = dict(sorted(split_stats.items()))
                total_stats.update(split_stats)

    report = {
        "dataset": DATASET,
        "subjects": list(args.subjects),
        "splits": list(args.splits),
        "question_only_output": str(question_only_path),
        "question_answer_output": str(question_answer_path),
        "report": str(report_path),
        "stats": dict(sorted(total_stats.items())),
        "per_subject_split": per_split,
        "errors": errors,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return total_stats


def main() -> int:
    args = parse_args()
    if args.page_size <= 0:
        print("--page-size must be positive", file=sys.stderr)
        return 2
    if args.limit_per_split is not None and args.limit_per_split <= 0:
        print("--limit-per-split must be positive when provided", file=sys.stderr)
        return 2
    if args.retries < 0:
        print("--retries must be zero or positive", file=sys.stderr)
        return 2

    stats = build_dataset(args)
    print(json.dumps(dict(sorted(stats.items())), ensure_ascii=False, indent=2))
    return 0 if stats.get("kept", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
