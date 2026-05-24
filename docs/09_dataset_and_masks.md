# Dataset 与 Loss Mask 学习笔记

## 数据层定位

`dataset/lm_dataset.py` 是训练链路的输入适配层。它把不同阶段的数据统一转换成训练脚本需要的 batch 字段：

- Pretrain：`input_ids, labels`
- SFT / LoRA / Distillation：`input_ids, labels`
- DPO：chosen/rejected 两组 `x/y/mask`
- PPO / GRPO：prompt 文本
- Agent RL：messages、tools、gt

这层最关键的价值是区分“模型看到的 token”和“loss 应该优化的 token”。Pretrain 基本对全文计算 loss；SFT、DPO 则只优化 assistant 输出区间。

## Chat 预处理

`pre_processing_chat` 对普通对话做概率性 system prompt 增强：

```text
如果首条不是 system，则以 add_system_ratio=0.2 的概率插入 system prompt
```

它内置中英文 system prompts，例如“你是 minimind，一个小巧但有用的语言模型”。这样做的意义是让模型在 SFT/RL 中见过更多系统提示形式，增强对 system role 的鲁棒性。

如果样本中含有 tools，函数会直接返回原 conversations，不做 system 增强，避免破坏工具调用数据的结构。

`post_processing_chat` 会以 80% 概率移除空 thinking 模板：

```text
<think>

</think>

```

只保留一部分空 think 样本，可以让模型既学会 thinking 格式，又避免所有回答都强制带空思考标签。

## PretrainDataset

`PretrainDataset` 读取 json 数据中的 `text` 字段：

```text
tokens = bos + tokenizer(text) + eos
input_ids = tokens + pad
labels = input_ids.clone()
labels[pad] = -100
```

这意味着预训练阶段对非 pad 的所有位置做 next-token prediction。模型学习的是普通语言建模能力，也就是“根据前文续写后文”。

默认会为文本显式加 `bos_token_id` 和 `eos_token_id`，并把长度控制在 `max_length` 内。

## SFTDataset

`SFTDataset` 读取 ShareGPT 风格 conversations，并通过 tokenizer 的 chat template 渲染成训练文本。

它支持：

- system/user/assistant/tool message
- system message 中携带 tools
- assistant message 中携带 tool_calls
- SFT 前后处理 system prompt 和 empty think

核心是 `generate_labels`。它先把全部 label 置为 `-100`，再扫描 token 序列中 assistant 起始和结束边界：

```text
bos_id = tokenizer("<|im_start|>assistant\n")
eos_id = tokenizer("<|im_end|>\n")
```

当找到 assistant 区间时，只把 assistant 内容到 `<|im_end|>\n` 的 token 设为训练目标。其他 system/user/tool prompt token 都保持 `-100`。

这就是 SFT 的 assistant-only loss：模型可以看到完整上下文，但只被优化去生成 assistant 回复。

## DPODataset

`DPODataset` 读取每条样本中的：

- `chosen`：一个 messages list
- `rejected`：一个 messages list

它分别对 chosen/rejected 应用 chat template，padding 到同一 `max_length`，再生成 loss mask。

返回字段是：

```text
x_chosen = chosen_input_ids[:-1]
y_chosen = chosen_input_ids[1:]
mask_chosen = chosen_loss_mask[1:]
x_rejected = rejected_input_ids[:-1]
y_rejected = rejected_input_ids[1:]
mask_rejected = rejected_loss_mask[1:]
```

DPO 训练脚本会把 chosen 和 rejected 拼接到同一个 batch 中。mask 仍然只覆盖 assistant 回复区间，因此 DPO 比较的是“回答部分”的 log probability，而不是 prompt 部分。

## RLAIFDataset

`RLAIFDataset` 用于 PPO/GRPO。它不返回 label，而是返回 prompt：

```text
prompt = apply_chat_template(conversations[:-1], add_generation_prompt=True)
```

也就是说，它去掉最后一条回答，只保留上下文，让 policy 在线生成 completion。`thinking_ratio` 控制是否在 generation prompt 中开启 thinking。

返回结构：

```text
{"prompt": prompt, "answer": ""}
```

当前 PPO/GRPO 的 reward 不是来自 Dataset 中的 answer，而是来自规则奖励和外部 reward model。

## AgentRLDataset

`AgentRLDataset` 手动读取 jsonl，每条样本返回：

```text
{
  "messages": messages_without_last_answer,
  "tools": tools,
  "gt": sample["gt"]
}
```

`parse_conversations` 会从 system message 中解析 tools，并去掉最后一条消息。训练时模型需要根据 messages 和 tools 做多轮 rollout，工具执行结果由 `train_agent.py` 回填。

`gt` 是工具任务的验证目标，用于 reward 计算：最终答案命中 gt 会加分，工具调用数量和参数正确性也会影响 reward。

## 各阶段 Mask 对比

| 阶段 | Dataset | 优化 token |
| --- | --- | --- |
| Pretrain | `PretrainDataset` | 全文非 pad token |
| Full SFT | `SFTDataset` | assistant 区间 |
| LoRA SFT | `SFTDataset` | assistant 区间 |
| Distillation | `SFTDataset` | assistant 区间 |
| DPO | `DPODataset` | chosen/rejected 的 assistant 区间 |
| PPO / GRPO | `RLAIFDataset` | rollout completion token |
| Agent RL | `AgentRLDataset` | 多轮 response token，工具观察 token 不直接优化 |

这个设计体现了训练阶段的差异：Pretrain 学文本分布，SFT/DPO 学 assistant 回答，RL 阶段则通过在线生成和 reward 优化策略。
