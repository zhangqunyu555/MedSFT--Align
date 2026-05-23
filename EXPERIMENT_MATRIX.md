# 实验对比矩阵

这份矩阵用于判断复现是否足以支撑简历描述。优先完成 P0，再推进 P1；P2 属于加分项。

## P0 必做

| 实验 | 对照组 | 实验组 | 核心结论 | 指标 |
| --- | --- | --- | --- | --- |
| Pretrain vs SFT | pretrain checkpoint | sft checkpoint | 预训练学习续写，SFT 学习对话格式和 assistant 响应 | PPL、对话格式正确率、生成样例 |
| RoPE ablation | 无 RoPE | 有 RoPE | 位置编码影响上下文顺序建模和长度泛化 | valid PPL、长文本 PPL、生成一致性 |
| MHA vs GQA | MHA | GQA | GQA 降低 KV-Cache 显存，推理更省 | KV cache MB/token、tokens/sec、PPL |
| LoRA rank | rank 8 | rank 16 / 32 | rank 越高容量越强但参数和显存增加 | 可训练参数量、PPL、SFT 样例评分 |
| SFT loss mask | 全量文本 loss | assistant-only loss | 对话 SFT 应避免让模型学习用户输入 | assistant loss、格式正确率、复读率 |
| KV-Cache | 关闭 cache | 开启 cache | cache 提升自回归生成速度 | decode tokens/sec、延迟、显存 |
| DPO | SFT model | DPO model | 偏好优化提升 chosen 相对 rejected 的概率 | preference accuracy、DPO loss、win rate |
| Dense vs MoE | Dense FFN | 4 experts Top-1 MoE | MoE 提升总容量，同时控制激活参数 | PPL、tokens/sec、显存、expert usage |

## P1 建议做

| 实验 | 对照组 | 实验组 | 核心结论 | 指标 |
| --- | --- | --- | --- | --- |
| warmup + cosine | constant lr | warmup + cosine | 调度器改善早期稳定性和最终收敛 | loss 波动、最终 PPL |
| bf16 vs fp16 | fp16 | bf16 | bf16 通常更稳定，适合 3090 训练链路复盘 | NaN 次数、loss 曲线、tokens/sec |
| 梯度累积 | 小 global batch | 大 global batch | 单卡用累积模拟更稳定 batch | loss 方差、吞吐、显存 |
| LoRA target modules | attention only | attention + FFN | FFN LoRA 可能带来更强任务适配能力 | 参数量、PPL、样例评分 |
| MoE load balance | 无 balance loss | 有 balance loss | 负载均衡降低 expert collapse | expert usage 方差、routing entropy |
| Tokenizer vocab size | 小词表 | 大词表 | 词表大小影响中文压缩率和训练效率 | tokens/sample、PPL、训练速度 |

## P2 加分项

| 实验 | 对照组 | 实验组 | 核心结论 | 指标 |
| --- | --- | --- | --- | --- |
| PPO / GRPO | SFT / DPO | PPO / GRPO | 在线 RL 能提升奖励目标，但需要控制 KL | reward、KL、格式正确率 |
| Reward Model | 规则评分 | learned RM | 学习式奖励可捕捉更复杂偏好，但有过拟合风险 | RM accuracy、偏好一致率 |
| YaRN / 长上下文 | 原始 RoPE | RoPE scaling | 长上下文外推能改善长文本 PPL | long-context PPL、生成稳定性 |
| Tool token SFT | 普通 SFT | tool-call SFT | 特殊 token 和模板让模型学会工具调用格式 | tool-call 格式正确率 |

## 记录模板

每个实验建议保存为 `reports/experiments/<experiment_name>.md`：

```text
实验名：
日期：
commit：
数据：
配置：
对照组：
实验组：
指标：
结论：
失败案例：
下一步：
```
