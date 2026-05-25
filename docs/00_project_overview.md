# MiniMind 复现项目总览

## 项目定位

本项目复现一条轻量级中文大模型训练链路：Tokenizer -> Pretrain -> Full SFT -> LoRA SFT -> DPO -> PPO / GRPO -> MoE。代码重点不是调用现成大模型框架训练，而是把 Decoder-Only Transformer、LoRA、偏好优化和强化学习训练流程拆开实现，方便理解每个模块在训练系统中的位置。

当前代码主要分成两层：

- `model/`：实现 MiniMind 模型结构、LoRA adapter 和 tokenizer 文件。
- `trainer/`：实现 tokenizer 训练、预训练、全参 SFT、LoRA SFT、DPO、PPO、GRPO、蒸馏、Agent RL 和 rollout 引擎。

`dataset/lm_dataset.py` 已补齐，负责把原始 json/jsonl 数据转换成各训练阶段需要的 batch 字段。它同时承担 chat template 渲染、assistant-only loss mask、DPO chosen/rejected mask、RLAIF prompt 构造和 Agent 工具数据解析。

## 训练阶段关系

`train_pretrain.py` 从随机初始化或指定权重开始训练基础语言模型，目标是 next-token prediction。它输出 `pretrain_<hidden_size>.pth`，作为后续 SFT 的起点。

`train_full_sft.py` 默认从 `pretrain` 权重继续训练，使用 SFT 数据让模型学习对话格式、指令跟随和 assistant 风格回答。它输出 `full_sft_<hidden_size>.pth`，是 LoRA、DPO、PPO、GRPO 的常用初始化权重。

`train_lora.py` 默认从 `full_sft` 权重开始，只训练注入到线性层中的 LoRA 参数。它用于说明参数高效微调：在冻结基座模型的情况下，用少量低秩参数适配新任务或新领域。

`train_dpo.py` 默认从 `full_sft` 权重初始化 policy model 和 reference model。policy model 训练，reference model 冻结，通过 chosen/rejected 偏好对让 policy 更倾向 chosen 回答。

`train_ppo.py` 和 `train_grpo.py` 都默认从 `full_sft` 权重开始，进入在线 rollout + reward 优化。PPO 使用 actor、critic、reference、reward model 四个角色；GRPO 使用组内多样本 reward 标准化，不需要 critic。

MoE 通过 `MiniMindConfig(use_moe=True)` 进入模型结构，训练脚本统一把 `res.aux_loss` 加到主 loss 上。因此同一套 pretrain、SFT、DPO、RL 入口可以复用到 Dense 和 MoE 两种模型。

## 工程主线

训练脚本基本遵循统一结构：

1. 初始化 DDP、随机种子和保存目录。
2. 构造 `MiniMindConfig`，按需读取 checkpoint/resume。
3. 设置 bf16/fp16 autocast 和 GradScaler。
4. 初始化模型、tokenizer、Dataset、DataLoader、optimizer。
5. 进入 epoch/step 循环，计算 loss，做梯度累积、裁剪、优化器更新。
6. 按间隔保存半精度权重和可恢复训练状态。

这种统一结构的意义是：不同训练阶段的差异集中在数据格式和 loss 计算，分布式、混合精度、checkpoint、日志等工程能力尽量复用。

## 当前文档结构

- `01_model_architecture.md`：模型结构、RoPE、GQA、SwiGLU、KV-Cache、MoE。
- `02_tokenizer.md`：ByteLevel BPE、特殊 token、chat template。
- `03_pretrain_and_sft.md`：预训练、全参 SFT 与训练工程组件。
- `04_lora_sft.md`：LoRA 实现、保存、加载、合并和 rank 对比。
- `05_dpo.md`：DPO 公式、policy/reference 关系和偏好优化意义。
- `06_rlhf_ppo_grpo.md`：PPO、GRPO、rollout、reward、KL 控制。
- `07_distillation_agent_and_moe.md`：蒸馏、Agent RL、MoE 扩展。
- `08_out_comparison_template.md`：后续 out 结果对比模板。
- `09_dataset_and_masks.md`：Dataset、chat 预处理和 loss mask。

## 实验问题复盘系列

- `experiment_notes/issues_and_lessons.md`：实验问题、解决方案、指标复盘和面试追问。
