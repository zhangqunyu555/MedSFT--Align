# Out 结果对比模板

这份模板用于接收后续 out 后处理结果。每组实验建议都写成“现象 -> 原因 -> 简历可表达结论”。

## 单次实验记录

| 项目 | 内容 |
| --- | --- |
| 实验名 | 待填 |
| 日期 | 待填 |
| Git commit | 待填 |
| 入口脚本 | 待填 |
| 训练命令 | 待填 |
| 权重来源 | 待填 |
| 输出权重 | 待填 |
| 数据集 | 待填 |
| hidden size / layers | 待填 |
| 是否 MoE | 待填 |
| batch size | 待填 |
| accumulation steps | 待填 |
| learning rate | 待填 |
| dtype | 待填 |
| GPU / 显存 | 待填 |
| 总耗时 | 待填 |

## 训练曲线

| step/epoch | loss | logits_loss | aux_loss | lr | reward | KL | avg_len | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 评测结果

| 指标 | 结果 | 说明 |
| --- | --- | --- |
| PPL | 待填 | 预训练/SFT 可用 |
| preference accuracy | 待填 | DPO 可用 |
| reward | 待填 | PPO/GRPO 可用 |
| KL_ref | 待填 | PPO/GRPO 可用 |
| tokens/sec | 待填 | 训练或推理吞吐 |
| peak memory | 待填 | 显存峰值 |
| format accuracy | 待填 | 对话/工具格式 |

## 样例输出

### Prompt

```text
待填
```

### Baseline Output

```text
待填
```

### Experiment Output

```text
待填
```

### 样例结论

待填。

## 对比实验：Pretrain vs SFT

| 项目 | Pretrain | Full SFT |
| --- | --- | --- |
| 权重 | `pretrain_<hidden_size>.pth` | `full_sft_<hidden_size>.pth` |
| 训练目标 | 文本续写 | 对话/指令跟随 |
| loss/PPL | avg_loss=4.6624, PPL=105.8949 | avg_loss=2.6158, PPL=13.6782 |
| 输出风格 | 更偏普通文本续写，不一定适应 user-assistant 问答格式 | 更能预测 assistant 回答内容，格式更像对话助手 |
| 结论 | 在 SFT-style 验证集上 PPL 较高，说明未充分适配问答格式 | PPL 显著下降，说明 SFT 有效提升对话格式和 assistant 回答建模能力 |

验证集规模：10 examples / 685 valid tokens。该结果适合作为流程验证和初步对比，正式指标建议扩展到 100-500 条固定 eval 样本。

简历可表达结论：在相同的 SFT 验证集上，Pretrain 模型的 PPL 为 105.89，而 Full SFT 模型的 PPL 降至 13.68，说明 SFT 显著提升了模型对指令问答格式的建模能力。Pretrain 阶段主要学习通用文本续写能力，对 user-assistant 对话格式不敏感；经过 SFT 后，模型能够更好地预测 assistant 回答内容，指令跟随和对话格式适配能力明显增强。

## 对比实验：Full SFT vs LoRA SFT

| 项目 | Full SFT | LoRA SFT |
| --- | --- | --- |
| 可训练参数量 | 全部参数 | 待填 |
| 参数占比 | 100% | 待填 |
| 显存 | 待填 | 待填 |
| loss/PPL | 待填 | 待填 |
| 样例效果 | 待填 | 待填 |
| 结论 | 待填 | 待填 |

简历可表达结论：待填。

## 对比实验：SFT vs DPO

| 项目 | SFT | DPO |
| --- | --- | --- |
| 权重来源 | `full_sft` | `full_sft` |
| 优化目标 | 模仿标准回答 | 提高 chosen 相对 rejected 偏好 |
| preference accuracy | 待填 | 待填 |
| 样例偏好 | 待填 | 待填 |
| 副作用 | 待填 | 待填 |

简历可表达结论：待填。

## 对比实验：PPO vs GRPO

| 项目 | PPO | GRPO |
| --- | --- | --- |
| 是否需要 critic | 是 | 否 |
| advantage 来源 | GAE | 组内 reward 标准化 |
| reward | 待填 | 待填 |
| KL | 待填 | 待填 |
| avg length | 待填 | 待填 |
| 稳定性 | 待填 | 待填 |
| 结论 | 待填 | 待填 |

简历可表达结论：待填。

## 对比实验：Dense vs MoE

| 项目 | Dense | MoE |
| --- | --- | --- |
| 总参数量 | 待填 | 待填 |
| 激活参数量 | 待填 | 待填 |
| aux loss | 不适用 | 待填 |
| expert usage | 不适用 | 待填 |
| loss/PPL | 待填 | 待填 |
| tokens/sec | 待填 | 待填 |
| 显存 | 待填 | 待填 |

简历可表达结论：待填。

## 对比实验：KV-Cache 开 / 关

| 项目 | 不使用 KV-Cache | 使用 KV-Cache |
| --- | --- | --- |
| 首 token 延迟 | 待填 | 待填 |
| decode tokens/sec | 待填 | 待填 |
| 显存占用 | 待填 | 待填 |
| 长上下文表现 | 待填 | 待填 |

简历可表达结论：待填。
