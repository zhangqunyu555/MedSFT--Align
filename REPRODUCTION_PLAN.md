# 基于 Qwen3 的中文医疗问答后训练与安全对齐复现思路

## 1. 项目目标

本项目围绕中文医疗问答场景，参考 [shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT) 的训练管线，复现一个基于 Qwen3 系列模型的领域后训练与安全对齐系统。整体目标覆盖医疗语料清洗、领域数据筛选、SFT / QLoRA、偏好或强化对齐、医疗评测、PPL 评估和错误案例分析。

目标基座模型选用 `Qwen3-4B-Instruct`。复现重点不是简单搬运 MedicalGPT，而是在其 PT、SFT、RM、PPO、DPO 等阶段设计基础上，将模型、数据筛选策略和评测指标适配到中文医疗问答与 C-eval 医学领域任务。

预期复现结果如下：

| 阶段 | 指标 | 目标结果 |
| --- | --- | --- |
| 原始 Qwen3-4B-Instruct | C-eval 医疗准确率 | 0.8324 |
| 高质量医疗 SFT / QLoRA 后 | C-eval 医疗准确率 | 0.8652 |
| PPO 强化对齐后 | C-eval 医疗准确率 | 0.8717 |
| SFT 前 | 1K 条专业医疗长文本 PPL | 15.194 |
| SFT 后 | 1K 条专业医疗长文本 PPL | 9.823 |
| PPO 前 | 复杂病例格式回答准确率 | 72% |
| PPO 后 | 复杂病例格式回答准确率 | 94% |

## 2. 参考基线与目录设计

MedicalGPT 提供了医疗大模型训练的完整参考路线，包括增量预训练、监督微调、奖励模型、PPO 强化学习、DPO 直接偏好优化等流程。复现时建议沿用其工程组织思想，将数据、训练、评测和脚本解耦，避免把实验逻辑写死在单个脚本中。

建议后续目录结构如下：

```text
medicalGPT-repro/
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

其中 `data/sft` 对齐 MedicalGPT 的监督微调数据组织方式，`data/preference` 对齐 `data/reward` 类偏好数据组织方式，`training` 存放 SFT、QLoRA、PPO、DPO 等训练入口，`evaluation` 存放 C-eval、PPL 和复杂病例格式评测逻辑，`analysis` 存放错误案例统计与人工复核结果。

## 3. 数据清洗与格式统一

原始数据包含约 50 万条中文医疗语料，来源可以是开源医疗问答数据集、医学科普问答、疾病咨询、药品问答、检查报告解释等。清洗阶段的目标是将异构语料统一为可训练、可追踪、可复现的 JSONL 数据。

### 3.1 字段校验

不同来源数据先映射为统一中间格式：

```json
{
  "id": "sample_id",
  "source": "dataset_name",
  "question": "用户问题",
  "answer": "医生或助手回答",
  "metadata": {
    "department": "科室",
    "disease": "疾病",
    "tags": ["标签"]
  }
}
```

字段校验规则：

- `question` 和 `answer` 必须存在且为非空字符串。
- 删除只有免责声明、广告、联系方式、乱码或无医学信息量的样本。
- 保留 `source`、`id`、`department` 等元信息，便于后续错误分析和数据回溯。
- 对多轮数据额外校验角色字段和轮次顺序。

### 3.2 去重与异常字符过滤

去重分为精确去重和近似去重：

- 精确去重：对标准化后的 `question + answer` 计算 hash。
- 近似去重：对问题或完整样本计算 SimHash / MinHash，删除高度重复样本。
- 同一问题存在多个答案时，优先保留更完整、更专业、无明显安全风险的回答。

异常字符处理包括：

- 过滤 HTML 标签、控制字符、不可见字符、重复空白。
- 统一中文标点、全角半角和换行格式。
- 删除含大量网址、电话、微信号、营销话术的样本。
- 对明显截断、拼接错误、编码错误的样本标记或删除。

### 3.3 长度过滤

根据 Qwen3-4B-Instruct 的训练上下文长度设置过滤阈值：

- 过短问题：少于 4 个中文字符的样本删除。
- 过短答案：少于 10 个中文字符的样本删除。
- 过长样本：超过训练最大长度的样本截断或删除，优先删除结构混乱的超长样本。
- 专业长文本保留独立评测集，不直接混入普通短问答评测。

### 3.4 多轮角色顺序修复

ShareGPT 多轮数据统一使用 `human` / `gpt` 角色：

- 首轮必须由 `human` 发起。
- 角色必须按 `human -> gpt -> human -> gpt` 交替。
- 连续同角色消息可合并，但需要保留原始语义顺序。
- 缺失助手回答或用户问题的轮次删除。
- system 信息如存在，单独保留为元信息或合并到首轮 prompt。

### 3.5 输出格式

Alpaca JSONL 用于单轮指令微调：

```json
{"instruction": "请回答以下医疗问题", "input": "高血压患者能否长期服用某药？", "output": "回答内容..."}
```

ShareGPT JSONL 用于多轮对话 SFT：

```json
{"conversations": [{"from": "human", "value": "问题内容"}, {"from": "gpt", "value": "回答内容"}]}
```

Preference JSONL 用于 DPO / 奖励模型 / PPO 数据构造：

```json
{"prompt": "复杂病例问题", "chosen": "更优回答", "rejected": "较差回答"}
```

或保留多轮上下文：

```json
{"conversations": [{"from": "human", "value": "病例问题"}], "chosen": "更优回答", "rejected": "较差回答"}
```

## 4. 领域数据筛选

用户给定的问题背景是：直接使用开源 Medical 数据集对 Qwen 模型做 SFT 后，C-eval 医学领域指标可能下降。核心原因通常包括训练数据和目标评测分布不一致、低质量样本干扰、问答风格与选择题评测能力冲突、医学事实噪声导致灾难性遗忘等。

因此本项目将 C-eval 医学测试集作为目标域分布参考，但不能将测试答案混入训练。具体筛选思路如下：

1. 准备约 50 万条原始中文医疗样本，清洗后得到候选池。
2. 准备 C-eval 医学领域测试题，仅使用题干、选项、科目等文本作为目标域语义描述。
3. 使用 embedding 模型对候选池和 C-eval 医学文本向量化。
4. 对每条候选样本计算其与 C-eval 医学集合的相似度。
5. 按最大相似度、Top-k 平均相似度或加权相似度排序。
6. 从候选池中筛选 10 万条高相似、高质量、覆盖均衡的数据，构建高质量 SFT 数据集。

embedding 模型可选：

- `bge-m3`
- `bge-large-zh-v1.5`
- Qwen embedding 系列模型

推荐的样本筛选评分：

```text
score(sample) = 0.7 * semantic_similarity_to_ceval
              + 0.2 * quality_score
              + 0.1 * diversity_score
```

其中：

- `semantic_similarity_to_ceval` 衡量与 C-eval 医学题目的语义接近程度。
- `quality_score` 衡量字段完整性、答案长度、专业术语密度、无广告和无乱码情况。
- `diversity_score` 控制科室、疾病、问题类型和答案形态的覆盖。

需要特别注意：C-eval 测试集只能用于目标域分布分析和相似度筛选，不允许将测试标签、标准答案或解析直接混入训练数据，否则会造成评测污染。

## 5. SFT / LoRA / QLoRA 指令微调

SFT 阶段以 `Qwen3-4B-Instruct` 为基座模型，使用筛选后的 10 万条高质量医疗 SFT 数据进行参数高效微调。优先采用 QLoRA，以降低显存占用；如果显存充足，可增加 LoRA 或全参数 SFT 对照实验。

建议训练配置沉淀到 `configs/sft_qwen3_4b_qlora.yaml`，包含：

- base model：`Qwen3-4B-Instruct`
- 训练数据：`data/sft/medical_sft_top100k.jsonl`
- 数据格式：Alpaca 或 ShareGPT
- 微调方式：QLoRA
- 量化：4-bit NF4
- 精度：bf16 或 fp16
- LoRA target modules：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`
- LoRA rank：建议从 8、16、32 做小规模对比
- learning rate：建议从 `1e-4` 到 `2e-4` 搜索
- batch size：按显存设置
- gradient accumulation：用于保持稳定的全局 batch size
- scheduler：cosine 或 linear warmup
- max sequence length：根据数据长度和显存设置
- checkpoint：按固定 step 保存，保留 best checkpoint 和 final checkpoint

训练完成后输出：

- LoRA / QLoRA adapter
- tokenizer 和训练配置快照
- 训练日志
- loss 曲线
- C-eval 医学验证结果
- 1K 医疗长文本 PPL 结果

验收标准：

- C-eval 医疗准确率从 0.8324 提升到约 0.8652。
- 1K 专业医疗长文本 PPL 从 15.194 降到约 9.823。
- 通用问答能力没有出现明显退化。
- 医疗回答不应出现更高比例的幻觉、过度诊断和危险用药建议。

## 6. 偏好数据与强化对齐

强化偏好对齐阶段的目标是提升复杂病例回答质量、答案结构稳定性和医疗安全性。数据来源为基于 SFT 数据集调用其他强模型构造的 5K 条复杂病例数据。

### 6.1 复杂病例数据构造

每条复杂病例建议包含：

- 主诉
- 现病史
- 既往史
- 检查指标
- 初步问题
- 期望回答要点
- 安全注意事项

模型输出格式建议统一为：

```text
1. 初步判断
2. 依据分析
3. 建议检查
4. 处理建议
5. 风险提示
6. 就医建议
```

构造数据时需要生成正负样本：

- `chosen`：结构完整、医学依据清楚、风险提示充分、避免绝对化诊断。
- `rejected`：格式缺失、依据不足、答非所问、忽略风险、存在危险建议或医学事实错误。

### 6.2 PPO 主线

PPO 使用 SFT 后模型作为 policy model，构造多维奖励函数：

```text
reward = w_format * reward_format
       + w_accuracy * reward_accuracy
       + w_similarity * reward_similarity
       + w_consistency * reward_consistency
       + w_safety * reward_safety
```

各奖励含义如下：

- `reward_format`：检查是否包含指定回答结构。
- `reward_accuracy`：使用标准要点或强模型评估医学准确性。
- `reward_similarity`：用 embedding 计算回答与参考答案的语义相似度。
- `reward_consistency`：检查推理过程和最终建议是否一致。
- `reward_safety`：惩罚危险用药、绝对化诊断、拒绝就医建议、伪造检查结果等问题。

PPO 阶段输出：

- PPO 后 adapter 或完整 checkpoint
- 奖励曲线
- KL 曲线
- 格式合规率曲线
- C-eval 医学结果
- 复杂病例人工或半自动评测结果

验收标准：

- C-eval 医疗准确率从 0.8652 提升到约 0.8717。
- 复杂病例格式回答准确率从 72% 提升到约 94%。
- KL 不应持续发散。
- 不能以牺牲医学准确性换取格式分数。

### 6.3 DPO 对照

DPO 作为 PPO 的稳定对照实验，使用同一批 5K preference 数据训练。DPO 不需要在线 rollout 和奖励模型，训练稳定性更好，适合在 PPO 前快速验证偏好数据质量。

DPO 验收重点：

- chosen / rejected 区分度是否足够。
- 格式和安全性是否提升。
- C-eval 医学准确率是否不下降。
- 与 PPO 相比是否存在更低成本、更稳定的收益。

## 7. 评测体系

评测阶段需要覆盖选择题医学能力、语言建模能力、复杂病例结构化回答能力和错误案例分析。

### 7.1 C-eval 医学领域评测

评测对象包括：

- 原始 `Qwen3-4B-Instruct`
- SFT / QLoRA 后模型
- DPO 后模型
- PPO 后模型

评测要求：

- 固定 prompt 模板。
- 固定 decoding 参数，选择题建议使用 greedy 或低温度。
- 保留每道题的预测选项、标准答案、模型输出和解析。
- 按医学子领域统计准确率。

### 7.2 1K 医疗长文本 PPL

准备 1K 条专业医疗长文本，作为领域语言建模能力评估集。该集合不参与训练。

评测要求：

- 使用同一 tokenizer。
- 使用相同 max length 和 stride 策略。
- 输出整体 PPL 和按文本类型分组的 PPL。
- 对 PPL 上升样本进行错误分析。

### 7.3 复杂病例格式与质量评测

评测维度：

- 是否包含固定回答模块。
- 医学结论是否谨慎。
- 是否给出合理检查和就医建议。
- 是否存在危险医疗建议。
- 回答是否和病例信息一致。

格式准确率可以由规则自动统计，医学准确性建议结合强模型评审和人工抽检。

### 7.4 错误案例分析

错误案例分类建议包括：

- 医学知识错误
- 选项理解错误
- 题干关键信息遗漏
- 过度诊断
- 用药风险
- 回答格式缺失
- 拒答或泛化回答
- 推理与结论不一致
- 训练数据噪声疑似影响

每轮训练后产出错误分析报告，重点观察新增错误和已修复错误，而不是只看整体分数。

## 8. 可复现实验配置

为了保证复现实验可追踪，所有阶段都需要固定以下信息：

- 代码版本或 commit hash
- Python、PyTorch、Transformers、PEFT、TRL、bitsandbytes 版本
- CUDA 和 GPU 型号
- 随机种子
- 数据快照路径和 hash
- 清洗规则版本
- embedding 模型版本
- 训练超参数
- checkpoint 路径
- 评测 prompt 模板
- decoding 参数

建议配置文件：

- `configs/data_cleaning.yaml`：清洗、去重、长度过滤、多轮修复规则。
- `configs/sft_qwen3_4b_qlora.yaml`：SFT / QLoRA 训练参数。
- `configs/ppo_qwen3_medical.yaml`：PPO 奖励函数、KL、rollout 和训练参数。
- `configs/eval_ceval_medical.yaml`：C-eval、PPL、复杂病例评测参数。

## 9. 风险与约束

医疗模型复现必须明确以下风险：

- 本项目仅用于科研和工程复现，不能替代医生诊断、治疗或用药建议。
- C-eval 测试集不得直接混入训练标签或参考答案，避免评测污染。
- embedding 筛选会让训练数据更贴近目标域，但可能降低数据多样性，需要保留多样性约束。
- SFT 可能导致模型过拟合医疗问答风格，并损伤通用能力，需要加入通用能力回归评测。
- PPO 奖励函数可能被模型利用，出现格式很好但事实错误的回答，需要医学准确性和安全性约束。
- 强模型生成的复杂病例数据需要抽检，避免把强模型幻觉蒸馏进目标模型。
- 医疗安全类回答必须避免绝对化诊断、危险用药建议和延误就医建议。
- 4B 模型能力有限，复杂医学推理能力提升应以评测证据为准，不能只依赖单个案例。

## 10. 阶段性里程碑

| 里程碑 | 输入 | 输出 | 验收 |
| --- | --- | --- | --- |
| M1 数据清洗 | 50 万原始医疗样本 | cleaned JSONL | 字段合法、重复率下降、异常样本减少 |
| M2 领域筛选 | cleaned JSONL + C-eval 医学文本 | 10 万高质量 SFT 样本 | 相似度高、覆盖均衡、无测试答案泄漏 |
| M3 SFT / QLoRA | 10 万 SFT 样本 + Qwen3-4B-Instruct | SFT adapter | C-eval 医疗准确率约 0.8652，PPL 约 9.823 |
| M4 偏好数据 | SFT 数据 + 强模型生成 | 5K preference / case 数据 | chosen / rejected 区分明确，格式统一 |
| M5 PPO / DPO | SFT adapter + preference 数据 | 对齐后模型 | C-eval 医疗准确率约 0.8717，格式准确率约 94% |
| M6 错误分析 | 全量评测输出 | 错误案例报告 | 明确主要错误类型和下一轮优化方向 |

## 11. 推荐实施顺序

1. 固定目录结构和配置文件模板。
2. 实现数据清洗脚本，输出 Alpaca、ShareGPT、Preference 三类 JSONL。
3. 实现 embedding 向量化和 C-eval 目标域相似度筛选。
4. 构建 10 万条高质量 SFT 数据集。
5. 使用 QLoRA 微调 `Qwen3-4B-Instruct`。
6. 跑 C-eval 医学评测和 1K 医疗长文本 PPL。
7. 构造 5K 条复杂病例 preference / PPO 数据。
8. 先跑 DPO 对照实验，再跑 PPO 强化对齐。
9. 汇总 C-eval、PPL、格式合规率和安全性指标。
10. 输出错误案例分析，并根据错误类型迭代数据筛选和奖励函数。

