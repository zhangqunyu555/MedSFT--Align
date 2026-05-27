# MedSFT-Align

MedSFT-Align 是一个面向中文医疗问答场景的领域后训练与安全对齐复现项目。项目目标是参考 [shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT) 的训练思路，基于 `Qwen3-4B-Instruct` 构建从医疗语料清洗、领域数据筛选、SFT / QLoRA 微调、偏好或强化对齐，到医疗评测与错误案例分析的完整流程。

详细复现设计见：[REPRODUCTION_PLAN.md](./REPRODUCTION_PLAN.md)。

阶段学习文档：

- [01 数据清洗](./docs/01_data_cleaning.md)
- [02 C-Eval 医学目标集](./docs/02_ceval_medical_target.md)
- [03 向量相似筛选](./docs/03_similarity_filtering.md)
- [04 shibing624/medical SFT 数据](./docs/04_shibing624_medical_sft_data.md)

## 项目目标

本项目计划完成以下内容：

- 构建中文医疗问答后训练数据处理流程。
- 将原始医疗语料统一清洗为 Alpaca、ShareGPT、Preference JSONL 格式。
- 基于 C-eval 医学领域测试集的语义分布，从约 50 万条原始医疗样本中筛选 10 万条高质量 SFT 样本。
- 对 `Qwen3-4B-Instruct` 进行 LoRA / QLoRA 指令微调。
- 构造 5K 条复杂病例偏好或强化学习数据。
- 使用 DPO 作为偏好对齐对照实验，使用 PPO 作为强化对齐主线。
- 建立 C-eval 医学准确率、医疗长文本 PPL、复杂病例格式准确率和错误案例分析评测体系。

## 预期指标

| 阶段 | 指标 | 目标结果 |
| --- | --- | --- |
| 原始 Qwen3-4B-Instruct | C-eval 医疗准确率 | 0.8324 |
| SFT / QLoRA 后 | C-eval 医疗准确率 | 0.8652 |
| PPO 强化对齐后 | C-eval 医疗准确率 | 0.8717 |
| SFT 前 | 1K 条专业医疗长文本 PPL | 15.194 |
| SFT 后 | 1K 条专业医疗长文本 PPL | 9.823 |
| PPO 前 | 复杂病例格式回答准确率 | 72% |
| PPO 后 | 复杂病例格式回答准确率 | 94% |

## 技术路线

### 1. 数据清洗与格式统一

对原始医疗语料进行字段校验、去重、异常字符过滤、长度过滤和多轮角色顺序修复。清洗后数据统一导出为：

- Alpaca JSONL：用于单轮指令微调。
- ShareGPT JSONL：用于多轮对话 SFT。
- Preference JSONL：用于 DPO、奖励模型或 PPO 数据构造。

### 2. 领域数据筛选

针对直接使用开源 Medical 数据集 SFT 后可能导致 C-eval 医学指标下降的问题，本项目将 C-eval 医学测试集作为目标域分布参考。使用 embedding 模型对原始医疗样本和 C-eval 医学题目进行语义向量化，并按语义相似度、数据质量和多样性筛选 10 万条高质量训练样本。

注意：C-eval 测试集仅用于目标域相似度分析，不能将测试答案、标签或解析混入训练数据。

### 3. SFT / QLoRA 指令微调

基于筛选后的 10 万条高质量医疗 SFT 数据，对 `Qwen3-4B-Instruct` 进行参数高效微调。训练中结合混合精度、梯度累积、学习率调度和 checkpoint 管理，沉淀可复现实验配置。

### 4. 偏好与强化对齐

调用强模型基于 SFT 数据构造 5K 条复杂病例数据，并设计多维奖励函数：

- 回答格式奖励
- 医学准确性奖励
- 语义相似度奖励
- 推理与结论一致性奖励
- 医疗安全奖励

训练阶段优先实现 PPO 强化对齐，同时保留 DPO 作为更稳定的偏好对齐对照实验。

### 5. 医疗评测与错误分析

评测体系包括：

- C-eval 医学领域准确率
- 1K 条专业医疗长文本 PPL
- 复杂病例格式回答准确率
- 医学事实错误、格式错误、安全风险、推理不一致等错误案例分类

## 建议目录结构

```text
MedSFT-Align/
├── README.md
├── REPRODUCTION_PLAN.md
├── configs/
│   ├── data_cleaning.yaml
│   ├── sft_qwen3_4b_qlora.yaml
│   ├── ppo_qwen3_medical.yaml
│   └── eval_ceval_medical.yaml
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── sft/
│   ├── preference/
│   └── eval/
├── scripts/
├── training/
├── evaluation/
└── analysis/
```

## 当前状态

- [x] 初始化项目仓库
- [x] 编写复现思路文档
- [x] 编写项目 README
- [ ] 搭建完整目录结构
- [x] 实现数据清洗脚本
- [x] 构建 C-Eval 医学双目标集脚本
- [x] 实现 embedding 筛选流程
- [x] 准备 shibing624/medical 50w 与 10w SFT 数据流程
- [ ] 实现 Qwen3 SFT / QLoRA 训练配置
- [ ] 实现 C-eval 与 PPL 评测
- [ ] 构造复杂病例偏好数据
- [ ] 实现 DPO / PPO 对齐实验
- [ ] 输出错误案例分析报告

## 免责声明

本项目仅用于科研复现和工程学习，不构成任何医学诊断、治疗或用药建议。模型输出不能替代专业医生意见。任何医疗相关结论都需要由具备资质的医疗专业人员审核。
