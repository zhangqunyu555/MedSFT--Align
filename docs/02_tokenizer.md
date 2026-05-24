# Tokenizer 学习笔记

## 当前定位

`trainer/train_tokenizer.py` 是一个学习和参考脚本。脚本开头明确说明不建议重复训练 tokenizer，因为 MiniMind 已自带词表。原因很现实：tokenizer 决定 token id 与文本片段的映射，如果不同训练者使用不同词表，模型权重就无法直接复用，输出也难以对齐。

当前仓库已有：

- `model/tokenizer.json`
- `model/tokenizer_config.json`

训练脚本默认从 `../model` 加载 tokenizer。

## ByteLevel BPE

脚本使用 `tokenizers` 库构建 BPE：

```text
Tokenizer(models.BPE())
pre_tokenizer = ByteLevel(add_prefix_space=False)
BpeTrainer(vocab_size=6400)
```

ByteLevel 的好处是可以覆盖任意 UTF-8 字符，不容易出现大量 unk。对中文小模型而言，6400 词表是一个折中：词表太小会让文本被切得很碎，序列更长；词表太大会增加 embedding/lm_head 参数量，也需要更多数据支撑。

## 特殊 token

脚本预留了 36 个特殊 token 位置，主要包括：

- 对话边界：`<|im_start|>`、`<|im_end|>`、`<|endoftext|>`
- 多模态预留：`<|vision_start|>`、`<|image_pad|>`、`<|audio_pad|>`、`<|video_pad|>`
- 工具调用：`<tool_call>`、`</tool_call>`、`<tool_response>`、`</tool_response>`
- 思考格式：`<think>`、`</think>`
- buffer token：`<|buffer1|>` 等

这些 token 的意义是把“结构”显式放进词表，而不是让模型用普通文本片段拼凑格式。对 SFT、RLHF、Agent RL 来说，稳定的边界 token 能显著降低格式学习难度。

## Chat Template

`tokenizer_config.json` 中写入了较完整的 chat template。它负责把 messages 转成统一文本格式：

```text
<|im_start|>system
...
<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
<think>
...
</think>

...
<|im_end|>
```

当传入 tools 时，模板会在 system 区域写入工具签名，并要求 assistant 用 `<tool_call>...</tool_call>` 返回 JSON 格式的工具调用。当 message role 为 `tool` 时，模板会把工具结果包装成 `<tool_response>...</tool_response>`。

这个模板把普通对话、thinking、tool call、tool response 都统一到同一种训练文本中，为后续 SFT 和 Agent RL 提供共同格式基础。

## 编码解码评估

`eval_tokenizer` 做了三类检查：

- chat template 渲染后编码再解码，检查是否一致。
- 中英文和混合文本压缩率，观察 chars/tokens。
- 流式解码时处理 ByteLevel 可能出现的半个 UTF-8 字符，避免输出乱码。

这些检查的意义是：tokenizer 不只是能训练出词表，还要保证训练文本、推理文本和流式输出都稳定。

## 与训练链路的关系

预训练阶段 tokenizer 把普通文本转成 next-token prediction 序列。SFT/DPO/RL 阶段 tokenizer 还承担 chat template 渲染职责，决定哪些文本属于 system、user、assistant 或 tool。

Dataset 层已经在 `dataset/lm_dataset.py` 中补齐。SFT 和 DPO 会扫描 chat template 渲染后的 token 序列，只把 `<|im_start|>assistant\n` 到 `<|im_end|>\n` 区间设为训练目标或偏好 mask；system/user/tool prompt token 只作为上下文，不直接参与 loss。
