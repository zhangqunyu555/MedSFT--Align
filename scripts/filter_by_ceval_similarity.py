#!/usr/bin/env python3
"""Filter medical SFT samples by embedding similarity to C-Eval medical targets."""

from __future__ import annotations

import argparse
import ast
import hashlib
import heapq
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_CONFIG: dict[str, Any] = {
    "input_path": "data/cleaned/cleaned_alpaca.jsonl",
    "target_path": "data/eval/ceval_medical_question_only.jsonl",
    "output_path": "data/sft/medical_sft_top100k_by_ceval_similarity.jsonl",
    "report_path": "data/sft/medical_sft_top100k_by_ceval_similarity_report.json",
    "embedding_backend": "transformers",
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "device": "auto",
    "batch_size": 32,
    "target_batch_size": 64,
    "max_length": 512,
    "top_k": 100000,
    "log_every": 500,
    "hash_embedding_dim": 384,
    "text_template": "指令：{instruction}\n问题：{input}\n回答：{output}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Top-K SFT samples by similarity to C-Eval medical targets."
    )
    parser.add_argument("--config", default="configs/similarity_filtering.yaml")
    parser.add_argument("--input", dest="input_path", default=None)
    parser.add_argument("--target", dest="target_path", default=None)
    parser.add_argument("--output", dest="output_path", default=None)
    parser.add_argument("--report", dest="report_path", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-backend", choices=["transformers", "hash"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
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
        "input_path": args.input_path,
        "target_path": args.target_path,
        "output_path": args.output_path,
        "report_path": args.report_path,
        "embedding_model": args.embedding_model,
        "embedding_backend": args.embedding_backend,
        "device": args.device,
        "batch_size": args.batch_size,
        "target_batch_size": args.target_batch_size,
        "max_length": args.max_length,
        "top_k": args.top_k,
        "log_every": args.log_every,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_rows(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


class TransformersEmbedder:
    def __init__(self, model_name: str, device: str, max_length: int) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def encode(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts
            embeddings = normalize_rows(embeddings)
        return embeddings.cpu()


class HashEmbedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> torch.Tensor:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token in text.split():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                col = value % self.dim
                sign = 1.0 if (value >> 8) & 1 else -1.0
                vectors[row_idx, col] += sign
            if not text.split():
                digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
                vectors[row_idx, int.from_bytes(digest, "little") % self.dim] = 1.0
        return normalize_rows(torch.from_numpy(vectors))


def build_embedder(config: dict[str, Any], device: str) -> Any:
    if config["embedding_backend"] == "hash":
        return HashEmbedder(int(config["hash_embedding_dim"]))
    return TransformersEmbedder(str(config["embedding_model"]), device, int(config["max_length"]))


def iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield record


def load_targets(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    targets = list(iter_jsonl(path))
    texts = [str(item.get("target_text", "")).strip() for item in targets]
    if not targets:
        raise ValueError(f"No target rows found: {path}")
    if any(not text for text in texts):
        raise ValueError(f"All target rows must contain non-empty target_text: {path}")
    return targets, texts


def encode_texts(embedder: Any, texts: list[str], batch_size: int) -> torch.Tensor:
    batches: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batches.append(embedder.encode(texts[start : start + batch_size]))
    return torch.cat(batches, dim=0)


def build_candidate_text(record: dict[str, Any], template: str) -> str:
    return template.format(
        instruction=str(record.get("instruction", "")).strip(),
        input=str(record.get("input", "")).strip(),
        output=str(record.get("output", "")).strip(),
    ).strip()


def update_heap(
    heap: list[tuple[float, int, dict[str, Any]]],
    top_k: int,
    record: dict[str, Any],
    score: float,
    sequence_id: int,
) -> None:
    item = (score, sequence_id, record)
    if len(heap) < top_k:
        heapq.heappush(heap, item)
        return
    if score > heap[0][0]:
        heapq.heapreplace(heap, item)


def enrich_record(
    record: dict[str, Any],
    score: float,
    best_target: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(record)
    enriched["similarity_score"] = round(float(score), 8)
    enriched["best_target_id"] = best_target.get("id")
    enriched["best_target_subject"] = best_target.get("subject")
    enriched["best_target_split"] = best_target.get("split")
    enriched["best_target_text"] = best_target.get("target_text")
    return enriched


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def filter_by_similarity(config: dict[str, Any]) -> dict[str, Any]:
    start_time = time.time()
    input_path = Path(config["input_path"])
    target_path = Path(config["target_path"])
    output_path = Path(config["output_path"])
    report_path = Path(config["report_path"])
    top_k = int(config["top_k"])
    batch_size = int(config["batch_size"])
    target_batch_size = int(config["target_batch_size"])
    device = resolve_device(str(config["device"]))

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if batch_size <= 0 or target_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target path does not exist: {target_path}")

    targets, target_texts = load_targets(target_path)
    embedder = build_embedder(config, device)
    target_embeddings = encode_texts(embedder, target_texts, target_batch_size)

    heap: list[tuple[float, int, dict[str, Any]]] = []
    scores_seen: list[float] = []
    total_read = 0
    encoded_count = 0
    skipped_empty_text = 0
    batches_done = 0
    batch_records: list[dict[str, Any]] = []
    batch_texts: list[str] = []
    sequence_id = 0

    def flush_batch() -> None:
        nonlocal encoded_count, sequence_id, batches_done
        if not batch_records:
            return
        candidate_embeddings = embedder.encode(batch_texts)
        score_matrix = candidate_embeddings @ target_embeddings.T
        best_scores, best_indices = score_matrix.max(dim=1)
        for local_idx, record in enumerate(batch_records):
            score = float(best_scores[local_idx].item())
            best_target = targets[int(best_indices[local_idx].item())]
            scores_seen.append(score)
            enriched = enrich_record(record, score, best_target)
            update_heap(heap, top_k, enriched, score, sequence_id)
            sequence_id += 1
            encoded_count += 1
        batches_done += 1
        log_every = int(config.get("log_every") or 0)
        if log_every > 0 and batches_done % log_every == 0:
            elapsed = time.time() - start_time
            print(
                f"progress: batches={batches_done} encoded={encoded_count} "
                f"heap={len(heap)} elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        batch_records.clear()
        batch_texts.clear()

    for record in iter_jsonl(input_path):
        total_read += 1
        text = build_candidate_text(record, str(config["text_template"]))
        if not text:
            skipped_empty_text += 1
            continue
        batch_records.append(record)
        batch_texts.append(text)
        if len(batch_records) >= batch_size:
            flush_batch()
    flush_batch()

    selected = [item[2] for item in sorted(heap, key=lambda row: row[0], reverse=True)]
    write_jsonl(output_path, selected)

    selected_scores = [float(item[0]) for item in heap]
    report = {
        "input_path": str(input_path),
        "target_path": str(target_path),
        "output_path": str(output_path),
        "report_path": str(report_path),
        "embedding_backend": config["embedding_backend"],
        "embedding_model": config["embedding_model"],
        "device": device,
        "batch_size": batch_size,
        "target_batch_size": target_batch_size,
        "log_every": int(config.get("log_every") or 0),
        "top_k": top_k,
        "target_count": len(targets),
        "total_read": total_read,
        "encoded_count": encoded_count,
        "skipped_empty_text": skipped_empty_text,
        "selected_count": len(selected),
        "score_mean_all": float(np.mean(scores_seen)) if scores_seen else None,
        "score_max_all": float(np.max(scores_seen)) if scores_seen else None,
        "score_min_selected": float(min(selected_scores)) if selected_scores else None,
        "score_max_selected": float(max(selected_scores)) if selected_scores else None,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    write_report(report_path, report)
    return report


def main() -> int:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    try:
        report = filter_by_similarity(config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
