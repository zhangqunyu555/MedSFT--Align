# PPO / GRPO 强化对齐学习笔记

## RLHF 训练入口

当前强化对齐主要在两个脚本中：

- `train_ppo.py`：PPO，包含 actor、critic、reference、reward model。
- `train_grpo.py`：GRPO/CISPO，使用组内 reward 标准化，不需要 critic。

二者都默认从 `full_sft` 权重开始，使用 `RLAIFDataset` 提供 prompt。`RLAIFDataset` 会取 `conversations[:-1]` 渲染 chat template，并设置 `add_generation_prompt=True`，让 policy 在线生成最后一轮 assistant completion。

## Rollout Engine

`trainer/rollout_engine.py` 抽象出两种 rollout 后端。

`TorchRolloutEngine` 直接调用当前 policy model 的 `generate`。优点是简单、无外部服务依赖；缺点是训练进程同时负责生成和反向传播，吞吐可能较低。

`SGLangRolloutEngine` 通过 HTTP 调用 SGLang 服务生成，并支持 `update_weights_from_disk` 更新服务端权重。优点是可以把 rollout 推理交给专门推理引擎，提高长回答和多样本生成效率；缺点是需要额外启动服务，并处理权重同步。

rollout 返回统一的 `RolloutResult`：

- `output_ids`：prompt + completion。
- `completion_ids`：completion 部分。
- `per_token_logps`：生成时旧策略的 token logprob。
- `completions`：解码后的文本。

这个抽象让 PPO、GRPO、Agent RL 可以复用同一套生成接口。

## Reward 设计

PPO 和 GRPO 的 reward 都包含：

- 长度奖励：回答长度在合理范围内加分，否则扣分。
- thinking 格式奖励：存在 `</think>`、思考内容长度适中、标签数量正确会加分。
- 重复惩罚：`rep_penalty` 用 n-gram 重复度扣分。
- 外部 reward model：`LMForRewardModel.get_score` 给回答质量打分，并裁剪到 `[-3, 3]`。

这是一种混合奖励：规则奖励保证格式和基本质量，reward model 提供语义偏好信号。

## PPO 角色与流程

`train_ppo.py` 中有四个模型角色：

| 角色 | 作用 |
| --- | --- |
| actor model | 当前可训练策略，负责生成回答 |
| critic model | 估计每个 token 的 value |
| reference model | 冻结基线，用于 KL 约束 |
| reward model | 给完整回答打外部奖励 |

PPO 的训练流程：

1. actor 对 prompt 做 rollout，得到 completion。
2. reward model 和规则函数计算每条回答的 reward。
3. actor 计算 old logprob，critic 计算 old value，reference 计算 ref logprob。
4. 使用 GAE 计算 advantage 和 return。
5. 多轮 PPO update，计算 clipped policy loss、value loss 和 KL penalty。
6. 如果 approx KL 超过阈值，触发 early stop，避免策略偏离过大。

PPO 的关键 loss 包括：

```text
ratio = exp(new_logp - old_logp)
policy_loss = max(-adv * ratio, -adv * clip(ratio))
value_loss = clipped value regression
total_loss = policy_loss + vf_coef * value_loss + aux_loss
```

另外代码还计算 reference KL penalty：

```text
exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1
```

它用于约束 policy 不要离 reference 太远。

## GRPO / CISPO 流程

GRPO 与 PPO 最大区别是没有 critic。它对每个 prompt 生成 `num_generations` 条回答，用组内 reward 标准化得到 advantage：

```text
advantage = (reward - group_mean) / (group_std + 1e-4)
```

这样同一个 prompt 下的多个回答互相比较，不需要 value model 估计 baseline。当前脚本默认 `num_generations=6`，默认 `loss_type=cispo`。

GRPO loss 逻辑：

- 计算当前 policy 每个 completion token 的 logprob。
- 计算 reference logprob 得到 per-token KL。
- 使用 old logprob 计算 ratio。
- 如果 `loss_type=grpo`，走 PPO-style clipped ratio。
- 如果 `loss_type=cispo`，使用上界截断 ratio 并 detach，限制大 ratio 对训练的影响。

GRPO 更轻量，适合小模型和算力有限场景；缺点是每个 prompt 需要多次生成，rollout 成本更高。

## PPO vs GRPO

| 项目 | PPO | GRPO |
| --- | --- | --- |
| 是否需要 critic | 需要 | 不需要 |
| advantage 来源 | GAE + value | 组内 reward 标准化 |
| 实现复杂度 | 高 | 中 |
| 训练稳定性 | 依赖 critic 质量 | 依赖组内 reward 方差 |
| 计算成本 | actor + critic 更新 | 多 generation rollout |
| 当前默认用途 | 完整 RLHF 复盘 | 轻量 RLAIF 实验 |

## 需要重点记录的曲线

后续 out 对比中，PPO/GRPO 至少记录：

- reward
- KL 或 KL_ref
- approx_kl
- clipfrac
- critic_loss，PPO 专有
- avg_response_len
- advantage mean/std
- policy_loss

这些曲线比单个样例更重要，因为 RL 训练很容易出现 reward 上升但 KL 发散、回答变长、格式投机等问题。
