# DPO 学习笔记

## DPO 的目标

DPO 用离线偏好数据训练模型，让 policy model 更倾向 chosen 回答，而不是 rejected 回答。它不需要在线 rollout，也不需要单独训练 reward model，因此比 PPO 更稳定、更容易复现。

当前 `train_dpo.py` 默认从 `full_sft` 权重开始，说明 DPO 被放在 SFT 之后，用来做偏好对齐。

## 模型角色

脚本初始化两个模型：

- policy model：可训练，最终保存为 `dpo_<hidden_size>.pth`。
- reference model：同样从 `full_sft` 加载，但 `eval()` 且 `requires_grad_(False)`。

reference model 的意义是提供一个“不被偏好训练改变的原始行为基线”。DPO 优化的不是 chosen 的绝对概率，而是 policy 相对 reference 在 chosen/rejected 上的偏好变化。

## Batch 组织

训练循环从 `DPODataset` 读取：

- `x_chosen`
- `x_rejected`
- `y_chosen`
- `y_rejected`
- `mask_chosen`
- `mask_rejected`

然后拼接成一个 batch：

```text
x = concat([x_chosen, x_rejected], dim=0)
y = concat([y_chosen, y_rejected], dim=0)
mask = concat([mask_chosen, mask_rejected], dim=0)
```

代码假设 batch 前半部分是 chosen，后半部分是 rejected。`DPODataset` 会分别对 chosen/rejected messages 应用 chat template，并用与 SFT 相同的 assistant 区间扫描逻辑生成 loss mask。最终 DPO 比较的是回答部分的 log probability，而不是 prompt 部分。

## Token log probability

`logits_to_log_probs` 先对 vocab 维做 `log_softmax`，再用 labels gather 出每个目标 token 的 log probability：

```text
log_probs = log_softmax(logits, dim=2)
log_probs_per_token = gather(log_probs, labels)
```

随后 `dpo_loss` 用 mask 只保留参与偏好优化的 token，并按序列求和：

```text
seq_log_prob = sum(token_log_prob * mask)
```

这一步把 token 级概率变成 response 级概率。

## DPO Loss

代码中的核心变量：

```text
pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
logits = pi_logratios - ref_logratios
loss = -logsigmoid(beta * logits)
```

直观理解：

- 如果 policy 比 reference 更偏向 chosen，`logits` 变大，loss 变小。
- 如果 policy 仍然偏向 rejected，或 chosen 优势不如 reference，loss 变大。
- `beta` 控制偏好优化强度。

这就是 DPO 的核心：不用显式 reward model，而是把偏好对转成“相对 reference 的 logprob margin”优化问题。

## 默认超参

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `from_weight` | `full_sft` | 从 SFT 模型开始偏好对齐 |
| `save_weight` | `dpo` | 输出权重前缀 |
| `learning_rate` | `4e-8` | 极低学习率，降低偏好训练导致遗忘的风险 |
| `beta` | `0.15` | DPO 偏好强度 |
| `batch_size` | 4 | chosen/rejected 拼接后实际前向 batch 翻倍 |
| `max_seq_len` | 1024 | 偏好样本最大长度 |

## 与 SFT 的区别

SFT 只告诉模型“这个回答应该怎么写”，目标是最大化标准答案 token 概率。DPO 同时看到 chosen 和 rejected，目标是让模型更偏好 chosen。它不是单纯模仿，而是学习排序。

后续 out 对比建议观察：

- DPO 前后 chosen/rejected logprob margin。
- DPO loss 是否下降。
- 样例中是否更偏向完整、安全、结构化回答。
- 是否出现过度对齐导致回答变短、拒答增多或基础能力下降。
