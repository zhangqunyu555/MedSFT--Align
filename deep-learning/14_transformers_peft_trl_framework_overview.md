# Transformers、PEFT、TRL 框架关系学习笔记

## 这篇文档解决什么问题

这篇文档专门回答一个面试里很容易被追问的问题：

```text
你项目里用的 Transformers、PEFT、TRL 是什么？它们是训练框架吗？
SFT、LoRA、DPO、PPO、GRPO 分别是用哪个框架实现的？
哪些是你自己实现的，哪些是调用开源框架？
```

一句话回答：

```text
Transformers、PEFT、TRL 都是 Hugging Face 生态里的训练与对齐工具库，但职责不同。
Transformers 负责模型、tokenizer、Trainer 和基础训练流程；
PEFT 负责 LoRA / QLoRA 这类参数高效微调；
TRL 负责 DPO、PPO、GRPO、RLOO 这类偏好优化和强化学习对齐训练器。
MedicalGPT 是我参考的项目框架，它把这些底层库组织成可运行的训练脚本。
```

所以更严谨地说，它们不完全是同一种“训练框架”：

| 名称 | 更准确的定位 | 在项目里的作用 |
| --- | --- | --- |
| `Transformers` | 模型训练与推理基础库 | 加载 Qwen3、加载 tokenizer、构造 `Trainer`、执行 SFT |
| `PEFT` | 参数高效微调库 | 实现 LoRA / QLoRA，只训练少量 adapter 参数 |
| `TRL` | 偏好优化与强化学习对齐库 | 实现 DPO、PPO、GRPO、RLOO 等对齐算法 |
| `bitsandbytes` | 量化与低精度计算库 | 支持 4bit / NF4 / QLoRA，降低显存 |
| `MedicalGPT` | 开源训练工程模板 | 提供 SFT、DPO、PPO、GRPO 等脚本组织方式 |

## 它们和我的项目是什么关系

MedSFT-Align 不是从零写一个大模型训练框架，而是在已有开源训练框架上复现并扩展一个中文医疗问答后训练流程。

项目核心链路是：

```text
Qwen3-4B-Instruct
  -> Transformers 加载模型和 tokenizer
  -> PEFT / bitsandbytes 做 QLoRA SFT
  -> TRL / 自定义 PPO 脚本做多维奖励强化对齐
  -> lm-evaluation-harness 做 C-Eval 评测
  -> 自写脚本做 PPL 和复杂病例格式准确率评测
```

在这个链路里：

- `Transformers` 是底座，几乎所有阶段都要用。
- `PEFT` 是微调方式，主要作用在 SFT、DPO、PPO、GRPO 的 LoRA adapter 上。
- `TRL` 是对齐算法库，主要作用在 DPO、PPO、GRPO 这些阶段。
- `MedicalGPT` 是参考工程，我基于它跑通训练流程，并补充了数据准备、筛选、格式转换、多维奖励 PPO 和评测脚本。

## Transformers：基础模型训练库

### 它是什么

`Transformers` 是 Hugging Face 的核心库，主要负责：

- 加载预训练模型：`AutoModelForCausalLM`
- 加载 tokenizer：`AutoTokenizer`
- 训练参数管理：`TrainingArguments`
- 标准训练循环：`Trainer`
- 文本生成：`model.generate()`
- 模型保存与加载：`save_pretrained()` / `from_pretrained()`

在我的项目中，Qwen3 模型就是通过 `Transformers` 加载的：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

### 在 SFT 里做什么

SFT 本质是让模型在给定 prompt 后学习标准答案。`Transformers` 负责最基础的监督训练：

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

这里的 `Trainer` 会完成：

- batch 构造；
- forward 计算；
- loss 反向传播；
- 梯度累积；
- 学习率调度；
- checkpoint 保存；
- evaluation；
- 日志记录。

### 面试怎么说

可以这样说：

```text
Transformers 在我项目里主要负责模型加载、tokenizer 加载、Qwen3 chat template 处理，以及 SFT 阶段的 Trainer 训练循环。它不是专门为 LoRA 或 RLHF 设计的，但它提供了所有训练的基础接口。
```

## PEFT：LoRA / QLoRA 参数高效微调库

### 它是什么

`PEFT` 全称是 Parameter-Efficient Fine-Tuning，意思是参数高效微调。

大模型全参训练显存和算力成本很高，PEFT 的思路是：

```text
冻结原始大模型参数，只训练很小一部分新增参数。
```

在我的项目里，主要用的是 LoRA / QLoRA。

LoRA 的核心思想是：不直接更新原始权重矩阵 `W`，而是在旁边加一个低秩增量：

```text
W' = W + ΔW
ΔW = B @ A
```

其中：

- `W` 是原始模型权重，冻结不训练；
- `A` 和 `B` 是低秩矩阵，可以训练；
- `r` 是 LoRA rank，控制低秩矩阵大小；
- 训练结束后只保存 adapter。

### 代码里怎么体现

典型 PEFT / LoRA 配置是：

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

这段代码做了几件事：

- 找到 Qwen3 里的线性层，比如 `q_proj / v_proj / o_proj`；
- 在这些线性层上注入 LoRA adapter；
- 冻结原始大模型参数；
- 只让 LoRA 参数参与训练。

### QLoRA 又是什么

QLoRA = Quantized LoRA。

它是在 LoRA 基础上，把 base model 用 4bit 方式加载，只训练 LoRA adapter：

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

直观理解：

```text
LoRA：冻结原模型，只训练小 adapter。
QLoRA：原模型还用 4bit 放进显存，进一步省显存。
```

### 面试怎么说

可以这样说：

```text
PEFT 在我项目里主要用于 LoRA / QLoRA 微调。Qwen3-4B 如果全参训练成本比较高，所以我使用 QLoRA：base model 4bit 量化加载，冻结主体参数，只训练 q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj 等模块上的 LoRA adapter。
```

## TRL：DPO / PPO / GRPO 对齐训练库

### 它是什么

`TRL` 全称通常理解为 Transformer Reinforcement Learning，是 Hugging Face 生态里做大模型偏好优化和强化学习对齐的库。

它提供的典型训练器包括：

- `DPOTrainer`
- `PPOTrainer`
- `GRPOTrainer`
- `RLOOTrainer`
- `RewardTrainer`

我的项目里，DPO、PPO、GRPO 这些训练学习文档都和 TRL 有关。

### DPO 里 TRL 做什么

DPO 输入是偏好数据：

```json
{
  "prompt": "问题",
  "chosen": "更好的回答",
  "rejected": "更差的回答"
}
```

TRL 的 `DPOTrainer` 会让模型对 `chosen` 的概率高于 `rejected`。

典型代码结构是：

```python
from trl import DPOConfig, DPOTrainer

training_args = DPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    learning_rate=5e-6,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

DPO 不需要在线采样，也不需要手写 reward function。它直接从 `chosen / rejected` 偏好对里学习。

### PPO 里 TRL 做什么

PPO 是强化学习对齐方法。它通常有四类组件：

```text
policy model：当前要更新的模型
reference model：冻结参考模型，用于 KL 约束
value model：估计状态价值，降低训练方差
reward model / reward function：给模型生成结果打分
```

我项目里为了复现简历里的“格式分 + 准确率分 + 安全分”，没有额外训练神经网络 reward model，而是实现了手写多维奖励函数：

```text
total_reward =
  0.30 * format_score
+ 0.50 * accuracy_score
+ 0.20 * safety_score
```

这里仍然可以做 PPO，因为 PPO 需要的是 reward signal，不一定必须是神经网络 reward model。reward 可以来自：

- 人类打分；
- 奖励模型；
- 规则函数；
- 工具调用结果；
- 评测器分数。

### GRPO 里 TRL 做什么

GRPO 更适合做规则奖励训练。它通常对同一个 prompt 采样多个回答，在组内比较相对好坏，不一定需要单独 value model。

在格式奖励、答案奖励这种任务里，GRPO 很自然：

```python
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[accuracy_reward, format_reward],
    args=training_args,
    train_dataset=train_dataset,
)
```

这里的 `reward_funcs` 可以直接是 Python 函数。

### 面试怎么说

可以这样说：

```text
TRL 在我项目里主要用于偏好对齐和强化学习训练。DPO 阶段使用 DPOTrainer 学 chosen/rejected 偏好；PPO 阶段我参考 TRL PPOTrainer 接口实现了 policy、reference、value 和 rule-based reward 四类组件；GRPO 文档里也分析了它为什么适合格式奖励和答案一致性奖励。
```

## MedicalGPT：是训练框架还是项目模板

`MedicalGPT` 更准确说不是一个底层训练框架，而是一个开源医学大模型训练项目模板。

它的价值是：

- 已经组织好了 `training/` 入口；
- 支持 SFT、DPO、PPO、GRPO 等训练脚本；
- 提供数据转换、adapter 合并、量化、评估等工具；
- 内部调用的是 `Transformers / PEFT / TRL` 这些库。

所以不能说：

```text
我自己从零实现了 SFT、LoRA、PPO 框架。
```

更准确应该说：

```text
我参考 MedicalGPT 的训练工程，使用 Transformers、PEFT、TRL 跑通了 Qwen3 医疗领域后训练流程，并自己实现了数据清洗、C-Eval 目标集构建、embedding 相似筛选、Qwen3 训练格式转换、多维奖励 PPO 数据构造、规则奖励函数、PPL 和复杂病例格式评测脚本。
```

这个表述既真实，也能体现工程能力。

## SFT、DPO、PPO、GRPO 分别对应哪些库

| 阶段 | 方法 | 主要依赖库 | 项目里谁在发挥作用 |
| --- | --- | --- | --- |
| SFT | 监督微调 | `Transformers` + `PEFT` + `bitsandbytes` | `Trainer` 训练，LoRA / QLoRA 降显存 |
| DPO | 直接偏好优化 | `TRL` + `PEFT` | `DPOTrainer` 学 chosen/rejected |
| PPO | 强化学习对齐 | `TRL` + 自定义 reward wrapper | PPO 更新 policy，规则函数给 reward |
| GRPO | 组相对策略优化 | `TRL` + reward functions | 多回答组内比较，适合规则奖励 |
| 评估 | C-Eval / PPL / 格式准确率 | `lm-evaluation-harness` + 自写脚本 | Benchmark 和项目指标计算 |

## 07-11 号文档里有没有讲这些框架

有，但讲法是分散在不同训练脚本里的。

### 07：SFT 文档

文件：

```text
deep-learning/07_sft_supervised_finetuning_code_reading.md
```

主要讲：

- `Transformers Trainer`
- `AutoModelForCausalLM`
- `AutoTokenizer`
- `TrainingArguments`
- `DataCollator`
- `PEFT`
- `LoRA / QLoRA`
- `bitsandbytes`
- `template_name qwen3`

这篇适合回答：

```text
SFT 是怎么训练的？
LoRA / QLoRA 怎么接进去？
为什么 labels 要 mask 用户输入？
```

### 08：DPO 文档

文件：

```text
deep-learning/08_dpo_training_code_reading.md
```

主要讲：

- `TRL DPOTrainer`
- `DPOConfig`
- `chosen / rejected` 偏好数据
- reference model
- DPO loss 的直观含义
- PEFT adapter 保存

这篇适合回答：

```text
DPO 和 SFT 有什么区别？
偏好数据怎么构造？
DPO 为什么不需要手写 reward model？
```

### 09：PPO / RLOO 文档

文件：

```text
deep-learning/09_ppo_training_code_reading.md
```

主要讲：

- MedicalGPT 原始 `ppo_training.py`
- TRL `RLOOTrainer`
- reward model
- policy / reference 关系
- 为什么这个脚本名字叫 PPO，但实现更接近 RLOO

这篇适合回答：

```text
MedicalGPT 原始 PPO 脚本是不是严格 PPO？
RLOO 和 PPO 有什么区别？
```

### 10：GRPO 文档

文件：

```text
deep-learning/10_grpo_training_code_reading.md
```

主要讲：

- `TRL GRPOTrainer`
- `GRPOConfig`
- `accuracy_reward`
- `format_reward`
- 多回答组内比较
- 为什么适合格式奖励

这篇适合回答：

```text
GRPO 为什么适合规则奖励？
格式奖励和准确率奖励怎么写？
```

### 11：自定义 PPO 多维奖励文档

文件：

```text
deep-learning/11_ppo_medical_multireward_pipeline_code_reading.md
```

主要讲：

- 自己实现的 `ppo_medical_multireward.py`
- `RuleBasedRewardModel`
- `ValueModelWrapper`
- `ValueScoreHead`
- `format_score / accuracy_score / safety_score`
- TRL experimental PPOTrainer 适配问题
- `.score`、`base_model_prefix`、dtype mismatch 等调试过程

这篇适合回答：

```text
你的 PPO 多维奖励函数怎么实现？
你没有 reward model 还能算 PPO 吗？
你亲手实现了哪些部分？
```

## 面试回答模板

### 问：Transformers、PEFT、TRL 是训练框架吗？

可以答：

```text
它们都属于 Hugging Face 生态里的训练工具库，但职责不同。Transformers 是基础模型训练和推理库，负责加载 Qwen3、tokenizer、Trainer 训练循环；PEFT 是参数高效微调库，负责 LoRA/QLoRA adapter；TRL 是偏好优化和强化学习对齐库，负责 DPO、PPO、GRPO 这类训练器。我的项目是基于 MedicalGPT 的工程结构，把这几个库组合起来完成中文医疗问答后训练。
```

### 问：你 SFT、LoRA、PPO 都是调包吗？

可以答：

```text
底层训练器我没有从零造轮子，SFT 使用 Transformers Trainer，LoRA/QLoRA 使用 PEFT，偏好和强化学习部分参考 TRL 和 MedicalGPT。这部分我会如实说是基于开源框架复现和改造。

我自己实现和重点工作的部分是数据链路和任务适配：50 万医疗数据准备、清洗规则、C-Eval 医学目标集构建、embedding 相似筛选 Top 10 万、Qwen3 训练格式转换、复杂病例 PPO 数据构造、格式分/准确率分/安全分多维奖励函数、PPL 评测和复杂病例格式准确率评测。
```

### 问：你为什么没有自己实现 Trainer / LoRA？

可以答：

```text
因为项目目标不是复现底层矩阵训练框架，而是复现一个医学领域后训练与安全对齐系统。底层训练框架使用成熟开源实现更稳，也更符合实际工程。我的工作重点是把医学数据、目标域筛选、Qwen3 模型、QLoRA 训练、PPO 奖励函数和评测指标串成可复现流程。
```

### 问：MedicalGPT、Transformers、PEFT、TRL 的关系是什么？

可以答：

```text
MedicalGPT 是上层工程项目，里面的训练脚本调用底层库。Transformers 负责模型和基础训练；PEFT 负责 LoRA/QLoRA；TRL 负责 DPO/PPO/GRPO 这类对齐算法。可以理解成 MedicalGPT 是项目模板，Transformers/PEFT/TRL 是它底层依赖的训练生态。
```

## 我的项目里哪些是开源复现，哪些是亲手实现

### 开源复现和调用的部分

- `Qwen3-4B-Instruct` 模型本身来自 Hugging Face / Qwen。
- SFT 训练主循环来自 MedicalGPT + Transformers。
- LoRA / QLoRA adapter 机制来自 PEFT。
- 4bit 量化来自 bitsandbytes。
- DPO / GRPO / PPO 训练器参考 TRL。
- C-Eval 评估使用 lm-evaluation-harness。

### 我亲手实现和改造的部分

- `shibing624/medical` 50 万候选数据准备脚本。
- 50 万语料清洗规则和清洗报告。
- C-Eval 临床医学 / 基础医学双目标集构建脚本。
- 50 万语料与 C-Eval 目标集 embedding 相似度筛选脚本。
- 10 万高相似 SFT 数据构造。
- Alpaca 到 ShareGPT / Qwen3 训练格式转换。
- 5K 复杂病例 PPO 数据集构造。
- `格式分 + 准确率分 + 安全分` 多维奖励函数。
- rule-based reward wrapper 和 value wrapper 的 PPO 适配。
- PPL 评测脚本。
- 复杂病例格式准确率评测脚本。
- SwanLab 训练记录脚本。
- 整体实验错误排查、指标整理和复盘文档。

## 最推荐背下来的项目描述

```text
这个项目不是从零实现一个大模型训练框架，而是基于 MedicalGPT、Transformers、PEFT 和 TRL 复现并改造了中文医疗问答后训练流程。

底层训练能力使用成熟开源库：Transformers 负责 Qwen3 模型加载和 SFT Trainer，PEFT 负责 LoRA/QLoRA，TRL 负责 DPO/PPO/GRPO 这类对齐训练器。

我自己重点实现的是医学任务适配和实验链路：包括 50 万医疗数据清洗、C-Eval 医学目标集构建、embedding 相似筛选 10 万高质量样本、Qwen3 训练格式转换、复杂病例 PPO 数据构造、多维规则奖励函数，以及 C-Eval、PPL、复杂病例格式准确率评测。
```

## 学习检查清单

面试前可以用下面这些问题检查自己是否真的懂了：

- 能不能说明 `Transformers` 和 `PEFT` 的区别？
- 能不能说明 `PEFT` 和 `bitsandbytes` 的区别？
- 能不能说明 `TRL` 和 `Transformers Trainer` 的区别？
- 能不能说明 SFT、DPO、PPO、GRPO 分别需要什么数据？
- 能不能说明为什么 LoRA 只训练 adapter？
- 能不能说明为什么 QLoRA 能省显存？
- 能不能说明为什么 DPO 不需要在线采样？
- 能不能说明为什么 PPO 可以使用手写 reward function？
- 能不能说明 MedicalGPT 原始 PPO 脚本为什么更接近 RLOO？
- 能不能说清楚自己亲手实现了哪些模块，而不是把开源框架说成自己写的？

