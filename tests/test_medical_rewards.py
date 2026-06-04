import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "MedicalGPT" / "training" / "ppo_medical_multireward.py"
SPEC = importlib.util.spec_from_file_location("ppo_medical_multireward", MODULE_PATH)
ppo_rewards = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ppo_rewards
SPEC.loader.exec_module(ppo_rewards)


def test_format_score_full_sections():
    text = "病情分析：胸痛需警惕。\n处理建议：完善心电图。\n风险提示：可能心梗。\n就医建议：及时就医。"
    assert ppo_rewards.compute_format_score(text, ["病情分析", "处理建议", "风险提示", "就医建议"]) == 1.0


def test_accuracy_keyword_score():
    score = ppo_rewards.compute_accuracy_score(
        "建议完善心电图和肌钙蛋白检查，警惕急性冠脉综合征。",
        reference_answer="胸痛患者需要心电图和肌钙蛋白检查。",
        answer_keywords=["心电图", "肌钙蛋白", "冠脉综合征"],
    )
    assert score > 0.6


def test_safety_penalizes_high_risk_without_doctor_hint():
    score = ppo_rewards.compute_safety_score("患者胸痛伴大汗怎么办？", "不用检查，在家休息即可。", "high")
    assert score < 0.7


def test_total_reward_contains_all_parts():
    record = {
        "prompt": "患者胸痛伴大汗怎么办？",
        "reference_answer": "需要警惕急性冠脉综合征，完善心电图和肌钙蛋白，及时就医。",
        "answer_keywords": ["急性冠脉综合征", "心电图", "肌钙蛋白"],
        "risk_level": "high",
        "required_sections": ["病情分析", "处理建议", "风险提示", "就医建议"],
    }
    response = "病情分析：需警惕急性冠脉综合征。处理建议：完善心电图和肌钙蛋白。风险提示：胸痛可能进展。就医建议：及时就医。"
    reward = ppo_rewards.compute_total_reward(record, response)
    assert set(reward) == {"format", "accuracy", "safety", "total"}
    assert reward["total"] > 0.7
