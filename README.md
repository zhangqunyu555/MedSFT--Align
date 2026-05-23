# MiniMind-64M Reproduction

本项目面向轻量级中文大模型训练场景，参考
[jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的 MiniMind 主线，从零复现一个约 64M 参数 Decoder-Only Transformer，并覆盖
Tokenizer、Pretrain、SFT、LoRA、DPO / RLHF 与 MoE 扩展的核心流程。

复现重点不是简单搬运代码，而是把关键模块、训练阶段、对比实验和评测证据整理成可解释、可运行、可复盘的工程项目。详细路线见
[REPRODUCTION_PLAN.md](./REPRODUCTION_PLAN.md)，实验矩阵见
[EXPERIMENT_MATRIX.md](./EXPERIMENT_MATRIX.md)。

## 项目目标

- 基于纯 PyTorch 实现 64M 参数 Decoder-Only Transformer，覆盖 RoPE、GQA、SwiGLU、RMSNorm、KV-Cache 等模块。
- 训练 BPE Tokenizer，完成从中文语料到模型输入的完整数据链路。
- 搭建 Pretrain、SFT、LoRA、DPO 训练入口，保留 PPO / GRPO / Reward Model 作为 RLHF 复盘和扩展模块。
- 扩展 4 experts / Top-1 Routing 的 MoE 版本，目标配置为约 198M 总参数 / 64M 激活参数。
- 结合 DDP、AMP / bf16、梯度累积、warmup + cosine 调度、KV-Cache 等工程优化，在单卡 RTX 3090 上完成可复现实验。
- 沉淀 loss、PPL、生成样例、吞吐、显存、KV-Cache 占用等对比结果，支撑简历中的技术叙述。

## 核心复现范围

| 模块 | 复现内容 | 验收证据 |
| --- | --- | --- |
| Tokenizer | BPE / ByteLevel 训练、特殊 token、chat template | 词表文件、编码解码测试、样例 token 分析 |
| Model | Decoder-Only Transformer、RoPE、GQA、SwiGLU、RMSNorm、KV-Cache | 参数量统计、前向单测、生成 smoke test |
| Pretrain | 中文文本续写训练 | train / valid loss、PPL 曲线、续写样例 |
| SFT | 对话格式训练、loss mask、chat template | 对话样例、SFT loss、格式正确率 |
| LoRA | 原生 LoRA 注入、rank 对比、权重合并 | 可训练参数占比、效果与显存对比 |
| DPO | chosen / rejected 偏好优化、冻结 reference model | preference accuracy、DPO loss、回答偏好变化 |
| RLHF | Reward Model、PPO / GRPO 流程复盘或最小实验 | 算法说明、最小训练日志、KL / reward 曲线 |
| MoE | 4 experts、Top-1 Routing、负载均衡 loss | routing 分布、负载均衡 loss、Dense vs MoE 对比 |

## 建议目录结构

```text
medicalGPT-repro/
├── README.md
├── REPRODUCTION_PLAN.md
├── EXPERIMENT_MATRIX.md
├── configs/
│   ├── tokenizer.yaml
│   ├── model_64m.yaml
│   ├── model_moe_198m_a64m.yaml
│   ├── pretrain.yaml
│   ├── sft.yaml
│   ├── lora.yaml
│   └── dpo.yaml
├── data/
│   ├── raw/
│   ├── tokenizer/
│   ├── pretrain/
│   ├── sft/
│   ├── preference/
│   └── eval/
├── minimind_repro/
│   ├── tokenizer/
│   ├── model/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── scripts/
├── tests/
├── runs/
└── reports/
```

## 当前状态

- [x] 初始化项目仓库
- [x] 明确 MiniMind-64M 复现目标
- [x] 补充实验对比矩阵
- [ ] 搭建目录结构与配置模板
- [ ] 实现 Tokenizer 训练与数据预处理
- [ ] 实现 64M Dense 模型
- [ ] 实现 Pretrain / SFT / LoRA / DPO 训练链路
- [ ] 实现 MoE 扩展
- [ ] 实现评测、曲线导出与实验报告
- [ ] 完成复现后合并到 `main`
