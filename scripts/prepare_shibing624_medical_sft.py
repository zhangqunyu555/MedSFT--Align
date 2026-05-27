#!/usr/bin/env python3
"""Prepare a fixed-size Alpaca SFT subset from shibing624/medical."""

from __future__ import annotations

import argparse
import ast
import codecs
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_name": "shibing624/medical",
    "subset": "finetune",
    "source_url": "https://huggingface.co/datasets/shibing624/medical/resolve/main/finetune/train_zh_0.json",
    "local_source_path": "",
    "sample_size": 500000,
    "strategy": "first",
    "seed": 42,
    "output_path": "data/raw/shibing624_medical/medical_zh_500k.jsonl",
    "report_path": "data/raw/shibing624_medical/medical_zh_500k_report.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed-size Alpaca JSONL subset from shibing624/medical."
    )
    parser.add_argument("--config", default="configs/shibing624_medical_500k.yaml")
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--local-source-path", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--strategy", choices=["first", "reservoir"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", dest="output_path", default=None)
    parser.add_argument("--report", dest="report_path", default=None)
    return parser.parse_args()


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return ""
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
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
        return value.strip("'\"")


def load_config(path: str) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config_path = Path(path)
    if not config_path.exists():
        return config

    parsed: dict[str, Any] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = parse_scalar(value.strip())
    config.update(parsed)
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overrides = {
        "source_url": args.source_url,
        "local_source_path": args.local_source_path,
        "sample_size": args.sample_size,
        "strategy": args.strategy,
        "seed": args.seed,
        "output_path": args.output_path,
        "report_path": args.report_path,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def open_source(config: dict[str, Any]) -> tuple[BinaryIO, str]:
    local_source = str(config.get("local_source_path") or "").strip()
    if local_source:
        path = Path(local_source)
        if not path.exists():
            raise FileNotFoundError(f"Local source does not exist: {path}")
        return path.open("rb"), str(path)

    source_url = str(config["source_url"]).strip()
    request = Request(source_url, headers={"User-Agent": "MedSFT-Align/shibing624-preparer"})
    return urlopen(request, timeout=120), source_url


class CharStream:
    """Small incremental UTF-8 character reader for large JSON sources."""

    def __init__(self, binary_stream: BinaryIO, chunk_size: int = 1024 * 1024) -> None:
        self.binary_stream = binary_stream
        self.chunk_size = chunk_size
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buffer = ""
        self.eof = False

    def fill(self) -> bool:
        if self.eof:
            return False
        chunk = self.binary_stream.read(self.chunk_size)
        if not chunk:
            self.buffer += self.decoder.decode(b"", final=True)
            self.eof = True
            return False
        self.buffer += self.decoder.decode(chunk)
        return True

    def ensure(self, min_chars: int = 1) -> bool:
        while len(self.buffer) < min_chars and not self.eof:
            self.fill()
        return len(self.buffer) >= min_chars

    def lstrip(self) -> None:
        while True:
            stripped = self.buffer.lstrip()
            if stripped or self.eof:
                self.buffer = stripped
                return
            self.buffer = ""
            self.fill()

    def pop(self, count: int = 1) -> str:
        self.ensure(count)
        value = self.buffer[:count]
        self.buffer = self.buffer[count:]
        return value


def iter_json_array(stream: BinaryIO) -> Iterable[dict[str, Any]]:
    reader = CharStream(stream)
    decoder = json.JSONDecoder()
    reader.lstrip()
    if not reader.ensure(1) or reader.pop(1) != "[":
        raise ValueError("Expected JSON array source")

    while True:
        reader.lstrip()
        if not reader.ensure(1):
            break
        if reader.buffer.startswith("]"):
            reader.pop(1)
            break
        if reader.buffer.startswith(","):
            reader.pop(1)
            reader.lstrip()

        while True:
            try:
                record, end = decoder.raw_decode(reader.buffer)
                reader.buffer = reader.buffer[end:]
                break
            except json.JSONDecodeError:
                if reader.eof:
                    raise
                reader.fill()
        if isinstance(record, dict):
            yield record


def iter_jsonl_from_prefix(stream: BinaryIO, prefix: bytes) -> Iterable[dict[str, Any]]:
    first_line = prefix + stream.readline()
    if first_line.strip():
        record = json.loads(first_line.decode("utf-8"))
        if isinstance(record, dict):
            yield record

    for raw_line in stream:
        if not raw_line.strip():
            continue
        record = json.loads(raw_line.decode("utf-8"))
        if isinstance(record, dict):
            yield record


def iter_source_records(stream: BinaryIO) -> Iterable[dict[str, Any]]:
    prefix = read_until_first_non_whitespace(stream)
    if not prefix:
        return
    if prefix[-1:] == b"[":
        prefixed = PrefixStream(prefix, stream)
        yield from iter_json_array(prefixed)
        return
    yield from iter_jsonl_from_prefix(stream, prefix)


class PrefixStream:
    """Binary-like stream that yields prefix bytes before the underlying stream."""

    def __init__(self, prefix: bytes, binary_stream: BinaryIO) -> None:
        self.prefix = prefix
        self.binary_stream = binary_stream

    def read(self, size: int = -1) -> bytes:
        if self.prefix:
            if size < 0 or size >= len(self.prefix):
                value = self.prefix
                self.prefix = b""
                if size < 0:
                    return value + self.binary_stream.read()
                return value
            value = self.prefix[:size]
            self.prefix = self.prefix[size:]
            return value
        return self.binary_stream.read(size)


def read_until_first_non_whitespace(stream: BinaryIO) -> bytes:
    prefix = bytearray()
    while True:
        char = stream.read(1)
        if not char:
            return bytes(prefix)
        prefix.extend(char)
        if not char.isspace():
            return bytes(prefix)


def normalize_record(record: dict[str, Any], row_id: int) -> tuple[dict[str, Any] | None, str | None]:
    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    output = str(record.get("output", "")).strip()

    if not instruction:
        instruction = "请回答以下医疗问题"
    if not output:
        return None, "missing_output"
    if not input_text and not instruction:
        return None, "missing_input"

    return (
        {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "source": "shibing624/medical",
            "row_id": row_id,
        },
        None,
    )


def write_jsonl_line(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_first(config: dict[str, Any]) -> Counter:
    sample_size = int(config["sample_size"])
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()

    stream, source = open_source(config)
    stats["source_opened"] = 1
    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            for row_id, raw_record in enumerate(iter_source_records(stream)):
                stats["total_seen"] += 1
                record, reason = normalize_record(raw_record, row_id)
                if reason:
                    stats[reason] += 1
                    continue
                write_jsonl_line(output_file, record)
                stats["kept"] += 1
                if stats["kept"] >= sample_size:
                    break
    finally:
        stream.close()
    stats["source"] = source
    return stats


def prepare_reservoir(config: dict[str, Any]) -> Counter:
    sample_size = int(config["sample_size"])
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(config["seed"]))
    reservoir: list[dict[str, Any]] = []
    stats: Counter = Counter()

    stream, source = open_source(config)
    stats["source_opened"] = 1
    try:
        for row_id, raw_record in enumerate(iter_source_records(stream)):
            stats["total_seen"] += 1
            record, reason = normalize_record(raw_record, row_id)
            if reason:
                stats[reason] += 1
                continue
            valid_index = stats["valid_seen"]
            stats["valid_seen"] += 1
            if len(reservoir) < sample_size:
                reservoir.append(record)
            else:
                replace_index = rng.randint(0, valid_index)
                if replace_index < sample_size:
                    reservoir[replace_index] = record
    finally:
        stream.close()

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in reservoir:
            write_jsonl_line(output_file, record)
            stats["kept"] += 1
    stats["source"] = source
    return stats


def write_report(config: dict[str, Any], stats: Counter, elapsed_seconds: float) -> None:
    report_path = Path(config["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_name": config["dataset_name"],
        "subset": config["subset"],
        "source_url": config["source_url"],
        "local_source_path": config.get("local_source_path") or "",
        "sample_size": int(config["sample_size"]),
        "strategy": config["strategy"],
        "seed": int(config["seed"]),
        "output_path": config["output_path"],
        "report_path": config["report_path"],
        "elapsed_seconds": round(elapsed_seconds, 3),
        "stats": dict(stats),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_dataset(config: dict[str, Any]) -> Counter:
    sample_size = int(config["sample_size"])
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    strategy = str(config["strategy"])
    if strategy == "first":
        stats = prepare_first(config)
    elif strategy == "reservoir":
        stats = prepare_reservoir(config)
    else:
        raise ValueError("strategy must be first or reservoir")

    if int(stats["kept"]) != sample_size:
        raise RuntimeError(
            f"Expected {sample_size} rows, but wrote {int(stats['kept'])}. "
            "The source may not contain enough valid Alpaca records."
        )
    return stats


def main() -> int:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    start = time.time()
    try:
        stats = prepare_dataset(config)
        write_report(config, stats, time.time() - start)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
