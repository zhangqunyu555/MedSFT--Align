#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a rule-reward PPO dataset from the medical SFT top-k JSONL.

The output is intentionally not a standard SFT dataset.  It keeps a prompt plus
reference metadata that a hand-written reward function can use during PPO.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REQUIRED_SECTIONS = ["病情分析", "处理建议", "风险提示", "就医建议"]

HIGH_RISK_TERMS = [
    "胸痛", "心梗", "心肌梗死", "卒中", "中风", "脑梗", "脑出血", "出血",
    "呼吸困难", "窒息", "昏迷", "休克", "抽搐", "惊厥", "高热", "感染性休克",
    "孕妇", "妊娠", "儿童", "婴儿", "新生儿", "老人", "老年", "肿瘤", "癌",
    "白血病", "糖尿病酮症", "过敏性休克", "自杀", "中毒", "急腹症",
]

MEDICAL_HINT_TERMS = [
    "诊断", "治疗", "检查", "症状", "病因", "并发症", "预防", "用药", "手术",
    "心电图", "肌钙蛋白", "血常规", "影像", "CT", "MRI", "感染", "炎症",
    "综合征", "梗死", "肿瘤", "贫血", "糖尿病", "高血压", "抗生素", "激素",
]

STOPWORDS = {
    "什么", "如何", "怎么", "哪些", "一种", "以及", "或者", "如果", "需要",
    "可以", "进行", "患者", "病人", "疾病", "导致", "描述", "以下", "下列",
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def clean_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_prompt(row: dict[str, Any]) -> str:
    instruction = clean_text(row.get("instruction"))
    input_text = clean_text(row.get("input"))
    if input_text:
        return f"{instruction}\n{input_text}" if instruction else input_text
    return instruction


def is_high_risk(prompt: str, answer: str) -> bool:
    text = f"{prompt}\n{answer}"
    return any(term.lower() in text.lower() for term in HIGH_RISK_TERMS)


def candidate_terms(text: str) -> list[str]:
    """Extract simple medical keyword candidates without external NLP packages."""
    chunks = re.split(r"[，。；;、,.：:\s（）()【】\[\]<>《》/\\]+", text)
    terms: list[str] = []
    for chunk in chunks:
        chunk = clean_text(chunk)
        if not chunk:
            continue
        if chunk in STOPWORDS:
            continue
        if len(chunk) < 2 or len(chunk) > 16:
            continue
        if re.fullmatch(r"\d+", chunk):
            continue
        terms.append(chunk)

    for hint in MEDICAL_HINT_TERMS + HIGH_RISK_TERMS:
        if hint in text and hint not in terms:
            terms.append(hint)

    counts = Counter(terms)
    ranked = sorted(counts, key=lambda item: (-counts[item], len(item)))
    return ranked[:8]


def convert_row(row: dict[str, Any], row_id: int) -> dict[str, Any] | None:
    prompt = build_prompt(row)
    reference_answer = clean_text(row.get("output"))
    if not prompt or not reference_answer:
        return None

    keywords = candidate_terms(reference_answer)
    if not keywords:
        keywords = candidate_terms(prompt)
    if not keywords:
        return None

    risk_level = "high" if is_high_risk(prompt, reference_answer) else "normal"
    return {
        "id": f"medical-ppo-{row_id}",
        "prompt": prompt,
        "reference_answer": reference_answer,
        "answer_keywords": keywords,
        "risk_level": risk_level,
        "required_sections": DEFAULT_REQUIRED_SECTIONS,
        "source": row.get("source", "shibing624/medical"),
        "source_row_id": row.get("row_id", row_id),
        "similarity_score": row.get("similarity_score"),
        "best_target_id": row.get("best_target_id"),
    }


def build_dataset(input_path: Path, sample_size: int, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_id, row in enumerate(iter_jsonl(input_path)):
        record = convert_row(row, row_id)
        if record is not None:
            records.append(record)

    if len(records) < sample_size:
        raise ValueError(f"Only {len(records)} valid records found, need {sample_size}.")

    high_risk = [r for r in records if r["risk_level"] == "high"]
    normal = [r for r in records if r["risk_level"] != "high"]
    rng = random.Random(seed)
    rng.shuffle(high_risk)
    rng.shuffle(normal)

    min_high = min(len(high_risk), int(sample_size * 0.30))
    selected = high_risk[:min_high]
    selected.extend(normal[: sample_size - len(selected)])

    if len(selected) < sample_size:
        selected.extend(high_risk[min_high: min_high + sample_size - len(selected)])

    selected = selected[:sample_size]
    rng.shuffle(selected)
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, rows: list[dict[str, Any]], input_path: Path) -> None:
    high_count = sum(1 for row in rows if row["risk_level"] == "high")
    report = {
        "input_path": str(input_path),
        "total_output": len(rows),
        "high_risk_count": high_count,
        "high_risk_ratio": round(high_count / len(rows), 4) if rows else 0,
        "required_sections": DEFAULT_REQUIRED_SECTIONS,
        "reward_weights": {"format": 0.30, "accuracy": 0.50, "safety": 0.20},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 5K medical PPO reward dataset from SFT top-k data.")
    parser.add_argument("--input", default="data/sft/shibing624_medical_top100k.jsonl")
    parser.add_argument("--output", default="data/rl/medical_complex_cases_5k.jsonl")
    parser.add_argument("--report", default="data/rl/medical_complex_cases_5k_report.json")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_dataset(Path(args.input), args.sample_size, args.seed)
    write_jsonl(Path(args.output), rows)
    write_report(Path(args.report), rows, Path(args.input))
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
