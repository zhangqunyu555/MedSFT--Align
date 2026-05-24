# 蒸馏、Agent RL 与 MoE 扩展学习笔记

## Knowledge Distillation

`train_distillation.py` 实现 teacher/student 蒸馏。默认配置可以用 MoE teacher 蒸馏 Dense student，也可以用更大 hidden size 的 teacher 蒸馏更小 student。

训练包含两类 loss：

```text
CE loss: student 对 ground-truth labels 的交叉熵
Distill loss: student logits 与 teacher logits 的 KL
total = alpha * CE + (1 - alpha) * Distill
```

`temperature` 用于软化 teacher 分布。温度越高，非最大 token 的概率信息越明显，student 能学到更丰富的“暗知识”。代码默认 `alpha=0.5`、`temperature=1.5`。

当前实现会对 label mask 有效位置计算蒸馏 loss。由于蒸馏脚本使用 `SFTDataset`，参与 CE 和 KL 的主要是 assistant 区间 token；system/user prompt 位置通过 `-100` mask 排除。

蒸馏的意义是：如果 MoE 或更大模型效果更好，可以把其输出分布迁移到更轻量模型上，尝试获得更好的小模型性价比。

## Agent RL

`train_agent.py` 在 GRPO/CISPO 风格训练基础上加入工具调用。它定义了一组模拟工具：

- `calculate_math`
- `unit_converter`
- `get_current_weather`
- `get_current_time`
- `get_exchange_rate`
- `translate_text`

工具调用使用 `<tool_call>...</tool_call>` 包裹 JSON，工具结果使用 `<tool_response>...</tool_response>` 回填给模型。`rollout_single` 支持多轮交互：模型生成工具调用，脚本解析并执行工具，再把工具结果追加到 messages 中继续生成。

这种设计让模型不仅学习“直接回答”，还学习“什么时候调用工具、如何组织工具参数、如何根据工具结果给最终答案”。

## Agent Reward

Agent reward 分两类。

如果没有工具调用，奖励类似普通 RLHF：

- 回答长度合理加分。
- thinking 格式正确加分。
- reward model 给回答质量打分。
- 重复内容扣分。

如果有工具调用，则奖励更关注工具使用是否正确：

- `<tool_call>` 标签数量不匹配会扣分。
- 工具名必须在可用工具列表中。
- 参数必须通过 `CHECK_ARGS` 校验。
- 工具调用数量与 ground truth 数量越接近越好。
- 最终答案中命中 ground truth 结果会加分。
- 多轮达到上限仍未完成会扣分。

这体现了 Agent RL 的训练重点：不是简单让回答更像人，而是让模型学会可靠地使用外部动作完成任务。

## MoE 扩展

MoE 在 `model/model_minimind.py` 中作为 FFN 替代实现。配置 `use_moe=True` 后，`MiniMindBlock` 会把 Dense `FeedForward` 换成 `MOEFeedForward`。

当前 MoE 关键机制：

- gate 对每个 token 输出 expert 概率。
- `torch.topk` 选出 top-k expert。
- 默认 top-1 routing，每个 token 只进入一个 expert。
- 每个 expert 内部仍是 SwiGLU FFN。
- 训练时计算 `aux_loss` 做负载均衡。

训练脚本统一把 `res.aux_loss` 加到主 loss 上，因此 Pretrain、SFT、DPO、PPO、GRPO 都可以直接支持 MoE。

## 总参数与激活参数

`trainer_utils.get_model_params` 会估算 total params 和 active params：

- total params：模型全部参数。
- active params：基础参数 + 每个 token 实际激活 expert 参数。

如果 active 小于 total，会打印：

```text
Model Params: <total>M-A<active>M
```

这对 MoE 简历表达很重要。MoE 的优势不是“每个 token 都用全部参数”，而是“总容量更大，但每个 token 只激活部分 expert”。

## Dense vs MoE 观察点

后续实验建议记录：

| 指标 | Dense | MoE |
| --- | --- | --- |
| 总参数量 | 待填 | 待填 |
| 激活参数量 | 待填 | 待填 |
| train loss | 待填 | 待填 |
| valid PPL | 待填 | 待填 |
| tokens/sec | 待填 | 待填 |
| 显存峰值 | 待填 | 待填 |
| expert usage | 不适用 | 待填 |
| aux loss | 不适用 | 待填 |

如果 MoE 出现某个 expert 占用过高，需要重点看负载均衡 loss、routing entropy 和 expert usage 分布。
