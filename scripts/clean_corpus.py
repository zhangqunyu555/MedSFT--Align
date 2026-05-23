#!/usr/bin/env python3
"""Clean Chinese medical QA corpora into Alpaca and ShareGPT JSONL files."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "input_path": "data/raw",
    "output_dir": "data/cleaned",
    "min_question_chars": 4,
    "min_answer_chars": 10,
    "max_question_chars": 2048,
    "max_answer_chars": 8192,
    "enable_dedup": True,
    "question_fields": ["question", "query", "prompt", "ask", "title"],
    "answer_fields": ["answer", "response", "output", "completion", "reply"],
    "instruction_fields": ["instruction"],
    "input_fields": ["input"],
    "output_fields": ["output"],
    "conversation_fields": ["conversations", "messages"],
    "user_roles": ["human", "user", "患者", "病人"],
    "assistant_roles": ["gpt", "assistant", "doctor", "医生", "客服"],
    "ad_keywords": [
        "加微信",
        "微信号",
        "联系电话",
        "免费咨询",
        "扫码",
        "推广",
        "代理",
        "私聊",
    ],
    "contact_patterns": [
        r"https?://\S+",
        r"www\.\S+",
        r"1[3-9]\d{9}",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        r"(微信|VX|vx|QQ|qq)[:：]?[A-Za-z0-9_-]{5,}",
    ],
}


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean JSONL medical QA data into Alpaca and ShareGPT JSONL."
    )
    parser.add_argument("--config", default="configs/data_cleaning.yaml")
    parser.add_argument("--input", dest="input_path", default=None)
    parser.add_argument("--output", dest="output_dir", default=None)
    return parser.parse_args()


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (SyntaxError, ValueError):
            return [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    return value.strip("'\"")


def load_config(path: str) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config_path = Path(path)
    if not config_path.exists():
        return config

    current_key: str | None = None
    parsed: dict[str, Any] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key:
            parsed.setdefault(current_key, []).append(parse_scalar(line[4:]))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_key = key
        parsed[key] = [] if value == "" else parse_scalar(value)

    config.update(parsed)
    return config


def iter_jsonl_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*.jsonl") if path.is_file())


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = SPACE_RE.sub(" ", text)
    text = "\n".join(part.strip() for part in text.splitlines())
    text = BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


def compact_for_dedup(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def count_content_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def pick_first(record: dict[str, Any], fields: list[str]) -> Any:
    for field in fields:
        if field in record and record[field] not in (None, ""):
            return record[field]
    return ""


def contains_ad_or_contact(text: str, config: dict[str, Any]) -> bool:
    if any(keyword and keyword in text for keyword in config["ad_keywords"]):
        return True
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in config["contact_patterns"])


def validate_lengths(question: str, answer: str, config: dict[str, Any]) -> str | None:
    q_len = count_content_chars(question)
    a_len = count_content_chars(answer)
    if q_len == 0 or a_len == 0:
        return "empty_field"
    if q_len < config["min_question_chars"] or a_len < config["min_answer_chars"]:
        return "too_short"
    if q_len > config["max_question_chars"] or a_len > config["max_answer_chars"]:
        return "too_long"
    return None


def normalize_role(role: Any, config: dict[str, Any]) -> str | None:
    role_text = str(role).strip()
    if role_text in config["user_roles"]:
        return "human"
    if role_text in config["assistant_roles"]:
        return "gpt"
    return None


def message_role(message: dict[str, Any]) -> Any:
    return message.get("from", message.get("role", ""))


def message_value(message: dict[str, Any]) -> Any:
    return message.get("value", message.get("content", ""))


def repair_conversations(
    conversations: Any, config: dict[str, Any]
) -> tuple[list[dict[str, str]] | None, bool]:
    if not isinstance(conversations, list):
        return None, False

    repaired: list[dict[str, str]] = []
    changed = False
    for message in conversations:
        if not isinstance(message, dict):
            changed = True
            continue
        role = normalize_role(message_role(message), config)
        value = normalize_text(message_value(message))
        if role is None or not value:
            changed = True
            continue
        if repaired and repaired[-1]["from"] == role:
            repaired[-1]["value"] = normalize_text(repaired[-1]["value"] + "\n" + value)
            changed = True
        else:
            repaired.append({"from": role, "value": value})

    while repaired and repaired[0]["from"] != "human":
        repaired.pop(0)
        changed = True
    while repaired and repaired[-1]["from"] != "gpt":
        repaired.pop()
        changed = True

    if len(repaired) < 2:
        return None, changed
    for index, message in enumerate(repaired):
        expected = "human" if index % 2 == 0 else "gpt"
        if message["from"] != expected:
            return None, True

    return repaired, changed


def extract_conversations(record: dict[str, Any], config: dict[str, Any]) -> Any:
    return pick_first(record, config["conversation_fields"])


def extract_single_turn(record: dict[str, Any], config: dict[str, Any]) -> tuple[str, str, str]:
    instruction = normalize_text(pick_first(record, config["instruction_fields"]))
    input_text = normalize_text(pick_first(record, config["input_fields"]))
    output = normalize_text(pick_first(record, config["output_fields"]))

    if instruction and output:
        question = normalize_text(f"{instruction}\n{input_text}" if input_text else instruction)
        return instruction, input_text, output

    question = normalize_text(pick_first(record, config["question_fields"]))
    answer = normalize_text(pick_first(record, config["answer_fields"]))
    return "请回答以下医疗问题", question, answer


def dedup_key_from_pair(question: str, answer: str) -> str:
    text = compact_for_dedup(question) + "\n" + compact_for_dedup(answer)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedup_key_from_messages(messages: list[dict[str, str]]) -> str:
    text = "\n".join(f"{item['from']}:{compact_for_dedup(item['value'])}" for item in messages)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def clean_corpus(config: dict[str, Any]) -> Counter:
    input_path = Path(config["input_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    files = iter_jsonl_files(input_path)
    report: Counter = Counter()
    report["input_files"] = len(files)

    seen: set[str] = set()
    alpaca_path = output_dir / "cleaned_alpaca.jsonl"
    sharegpt_path = output_dir / "cleaned_sharegpt.jsonl"

    with alpaca_path.open("w", encoding="utf-8") as alpaca_out, sharegpt_path.open(
        "w", encoding="utf-8"
    ) as sharegpt_out:
        for file_path in files:
            with file_path.open("r", encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, start=1):
                    if not line.strip():
                        continue
                    report["total_read"] += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        report["invalid_json"] += 1
                        continue
                    if not isinstance(record, dict):
                        report["invalid_record"] += 1
                        continue

                    conversations = extract_conversations(record, config)
                    if conversations:
                        messages, changed = repair_conversations(conversations, config)
                        if changed:
                            report["multi_turn_repaired"] += 1
                        if not messages:
                            report["multi_turn_failed"] += 1
                            continue

                        text_for_filter = "\n".join(item["value"] for item in messages)
                        if contains_ad_or_contact(text_for_filter, config):
                            report["ad_or_contact"] += 1
                            continue

                        key = dedup_key_from_messages(messages)
                        if config["enable_dedup"] and key in seen:
                            report["duplicate"] += 1
                            continue
                        seen.add(key)

                        write_jsonl(sharegpt_out, {"conversations": messages})
                        report["sharegpt_kept"] += 1
                        report["kept"] += 1
                        continue

                    instruction, input_text, answer = extract_single_turn(record, config)
                    question = normalize_text(f"{instruction}\n{input_text}" if input_text else input_text)
                    question_for_validation = normalize_text(input_text or instruction)
                    if instruction == "请回答以下医疗问题":
                        question = input_text
                        question_for_validation = input_text

                    reason = validate_lengths(question_for_validation, answer, config)
                    if reason:
                        report[reason] += 1
                        continue
                    if contains_ad_or_contact(question + "\n" + answer, config):
                        report["ad_or_contact"] += 1
                        continue

                    key = dedup_key_from_pair(question_for_validation, answer)
                    if config["enable_dedup"] and key in seen:
                        report["duplicate"] += 1
                        continue
                    seen.add(key)

                    write_jsonl(
                        alpaca_out,
                        {"instruction": instruction, "input": input_text, "output": answer},
                    )
                    report["alpaca_kept"] += 1
                    report["kept"] += 1

    report_path = output_dir / "cleaning_report.json"
    report_data = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "alpaca_output": str(alpaca_path),
        "sharegpt_output": str(sharegpt_path),
        "stats": dict(sorted(report.items())),
    }
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.input_path:
        config["input_path"] = args.input_path
    if args.output_dir:
        config["output_dir"] = args.output_dir

    input_path = Path(config["input_path"])
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        return 2

    report = clean_corpus(config)
    print(json.dumps(dict(sorted(report.items())), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
