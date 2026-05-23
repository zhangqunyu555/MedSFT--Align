# MiniMind-64M 从零训练与后训练复现计划

## 1. 定位

本项目参考 MiniMind 的教育型小模型训练路线，目标是从零构建一套轻量级中文大模型训练与后训练系统。最终成果需要同时满足两个要求：

- 工程上能跑通 Tokenizer -> Pretrain -> SFT -> LoRA -> DPO / RLHF -> MoE 的核心链路。
- 表达上能支撑简历叙述，说明每个模块为什么做、对训练或推理有什么影响、用什么实验数据证明。

MiniMind 最新主线已经覆盖 Dense 64M、MoE 198M-A64M、BPE + ByteLevel Tokenizer、SFT、LoRA、DPO、PPO / GRPO / CISPO、Tool Use 等内容。本项目优先复现主线能力，再选择性保留 Tool Use、Agentic RL、长上下文外推等扩展为加分项。

## 2. 里程碑

| 阶段 | 输入 | 输出 | 验收 |
| --- | --- | --- | --- |
| M0 工程骨架 | 当前空仓库 | 目录、配置、依赖、测试框架 | smoke test 可运行 |
| M1 Tokenizer | 中文预训练语料 | BPE tokenizer、特殊 token、chat template | 编码解码一致，覆盖中英文和模板 token |
| M2 Dense 模型 | model config | 64M Decoder-Only Transformer | 参数量正确，forward / generate 通过 |
| M3 Pretrain | 预训练文本 | base checkpoint | loss / PPL 曲线下降，续写样例可读 |
| M4 SFT | 对话数据 | SFT checkpoint | 对话格式稳定，loss mask 正确 |
| M5 LoRA | SFT 数据 | LoRA adapter / merged model | rank 8 / 16 / 32 对比完成 |
| M6 DPO | preference 数据 | DPO checkpoint | chosen 胜率提升，reference model 冻结正确 |
| M7 RLHF 复盘 | preference 或 reward 数据 | RM + PPO / GRPO 最小实现或报告 | reward / KL 曲线可解释 |
| M8 MoE | pretrain / sft 数据 | 198M-A64M MoE checkpoint | routing 分布可视化，负载均衡有效 |
| M9 报告与合并 | 全部实验日志 | reports 与 README 更新 | 通过测试后合并 `main` |

## 3. 核心实现

### 3.1 Tokenizer

- 使用 BPE / ByteLevel 训练中文小模型 tokenizer。
- 保留 `<|im_start|>`、`<|im_end|>`、`<think>`、`<tool_call>`、`<tool_response>` 等可扩展特殊 token。
- 实现 chat template，将 SFT / preference 数据统一渲染为模型训练文本。
- 增加 tokenizer smoke test：中文、英文、数字、医学术语、代码片段、特殊 token 往返编码。

### 3.2 Dense 64M 模型

模型使用纯 PyTorch 实现，重点模块如下：

- Decoder-Only Transformer
- RMSNorm
- RoPE
- MHA / GQA 可切换注意力
- SwiGLU FFN
- Causal mask
- KV-Cache
- tied embedding 可配置

必须提供参数量统计脚本，分别输出 embedding、attention、FFN、norm、lm_head 等模块参数量，避免只写“64M”但无法解释参数来源。

### 3.3 训练链路

Pretrain 阶段验证模型的续写能力，SFT 阶段验证模型的对话能力，两者需要明确区分：

- Pretrain：对完整文本做 next-token prediction，目标是语言建模。
- SFT：只对 assistant 部分计算 loss，目标是学习指令和对话格式。

训练入口应支持：

- bf16 / fp16 AMP
- 梯度累积
- DDP
- gradient clipping
- warmup + cosine scheduler
- checkpoint 保存与恢复
- train / valid loss、PPL、tokens/sec、显存统计

### 3.4 LoRA

优先实现原生 LoRA，目标模块至少包括：

- attention: `q_proj`、`k_proj`、`v_proj`、`o_proj`
- FFN: `gate_proj`、`up_proj`、`down_proj`

必须记录：

- rank
- alpha
- dropout
- 可训练参数量与占比
- 显存占用
- 训练速度
- SFT / eval 效果

### 3.5 DPO 与 RLHF

DPO 是偏好对齐主线的优先落地点，因为它不需要在线 rollout，适合小模型复现。实现要求：

- policy model 可训练
- reference model 冻结
- chosen / rejected log probability 计算正确
- beta 可配置
- 输出 DPO loss、chosen reward、rejected reward、preference accuracy

RLHF 部分建议分两层：

- 必做：复盘 Reward Model、PPO、GRPO 的完整范式，并保留接口或最小实现。
- 加分：跑通一个小规模 PPO / GRPO 实验，记录 reward、KL、response length、格式正确率曲线。

### 3.6 MoE

MoE 版本以 4 experts / Top-1 Routing 为优先目标：

- Dense FFN 替换为 MoE FFN。
- 每个 token 只激活一个 expert。
- 记录总参数量和激活参数量。
- 引入负载均衡 loss，降低路由集中。
- 输出 expert usage 分布、routing entropy、负载均衡 loss 曲线。

## 4. 必做对比实验

你已有的 4 组对比是主线：

| 对比 | 目的 |
| --- | --- |
| Pretrain vs SFT | 说明预训练是续写，SFT 是对话 |
| 无 RoPE vs 有 RoPE | 说明位置编码对长短上下文建模的作用 |
| MHA vs GQA | 说明推理 KV-Cache 显存差异 |
| LoRA 不同 rank | 说明参数效率和效果的权衡 |

建议再补充以下对比，使项目更像完整训练系统，而不只是模块展示：

| 对比 | 为什么值得做 | 推荐指标 |
| --- | --- | --- |
| SFT loss mask 开 / 关 | 直接体现对话训练和普通 LM 训练的差异 | assistant loss、格式正确率、无关复读率 |
| bf16 vs fp16 / fp32 | 说明混合精度对稳定性、速度、显存的影响 | loss 曲线、tokens/sec、显存峰值、NaN 次数 |
| warmup + cosine vs constant lr | 体现训练稳定性工程经验 | early loss 波动、最终 PPL |
| batch size / 梯度累积对比 | 说明单卡 3090 下如何获得稳定全局 batch | loss 方差、吞吐、显存 |
| KV-Cache 开 / 关 | 直观说明推理优化收益 | 首 token 延迟、生成 tokens/sec、显存 |
| Dense FFN vs MoE FFN | 支撑 198M 总参数 / 64M 激活参数叙述 | PPL、tokens/sec、expert usage、显存 |
| MoE 无负载均衡 vs 有负载均衡 | 说明 routing collapse 与 load balance loss | expert usage 方差、routing entropy、loss |
| DPO 前 vs DPO 后 | 证明偏好优化确实改变回答偏好 | preference accuracy、win rate、样例对比 |
| LoRA target modules 对比 | 区分只训 attention 与 attention + FFN 的收益 | 可训练参数量、PPL、对话评分 |
| Tokenizer 词表大小对比 | 说明中文小模型 tokenizer 粒度影响 | 平均 tokens/样本、训练速度、PPL |

## 5. 评测与报告

每个实验至少保留以下产物：

- config 快照
- git commit hash
- 数据快照或数据 hash
- 训练日志
- loss / PPL 曲线
- 推理样例
- 显存与吞吐统计
- 简短结论：这个实验说明了什么

最终报告建议分为：

- `reports/model_architecture.md`
- `reports/tokenizer.md`
- `reports/pretrain_vs_sft.md`
- `reports/lora.md`
- `reports/dpo_rlhf.md`
- `reports/moe.md`
- `reports/final_summary.md`

## 6. 合并策略

当前开发应在功能分支完成，复现完成后再合并到 `main`。合并前需要满足：

- 单元测试通过。
- 至少一个端到端 smoke test 通过。
- README、复现计划、实验矩阵和最终报告同步更新。
- 大文件、checkpoint、数据集不直接提交到 Git，改用下载脚本或路径说明。
- `main` 合并前先同步最新主线，解决冲突后再合并。
