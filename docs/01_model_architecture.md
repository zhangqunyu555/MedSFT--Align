# 模型架构学习笔记

## 配置层：MiniMindConfig

`MiniMindConfig` 继承自 Hugging Face 的 `PretrainedConfig`，让自定义模型能保存、加载并接入 `PreTrainedModel` 生态。核心配置包括：

| 配置 | 作用 |
| --- | --- |
| `hidden_size` | Transformer 隐状态维度，默认 768 |
| `num_hidden_layers` | Decoder block 层数，默认 8 |
| `num_attention_heads` | query head 数，默认 8 |
| `num_key_value_heads` | key/value head 数，默认 4，用于 GQA |
| `head_dim` | 每个 head 的维度，默认 `hidden_size / num_attention_heads` |
| `intermediate_size` | FFN 中间层维度，默认按 `hidden_size * pi` 近似并对齐到 64 |
| `max_position_embeddings` | RoPE 预计算最大长度，默认 32768 |
| `rope_theta` | RoPE base，默认 `1e6` |
| `inference_rope_scaling` | 是否启用 YaRN 风格 RoPE scaling |
| `use_moe` | 是否把 FFN 替换为 MoE FFN |

这套配置的设计思路是：Dense 和 MoE 共用同一个模型入口，只通过 `use_moe` 等参数切换结构；MHA/GQA 也通过 `num_key_value_heads` 控制。

## RMSNorm

`RMSNorm` 使用均方根归一化：

```text
x_norm = x / sqrt(mean(x^2) + eps)
out = weight * x_norm
```

它不减均值，只缩放激活幅度。相比 LayerNorm，RMSNorm 更简单，常见于 LLaMA/Qwen 类 Decoder-only 模型。代码里 attention 前和 FFN 前各有一个 RMSNorm，属于 pre-norm 结构，有助于深层训练稳定。

## RoPE 与 YaRN

`precompute_freqs_cis` 预计算 cos/sin，`apply_rotary_pos_emb` 把旋转位置编码作用到 q/k 上。RoPE 的意义是把相对位置信息写进注意力内积路径，让模型理解 token 顺序，而不是给 embedding 直接加绝对位置向量。

代码还保留了 `inference_rope_scaling` 分支：当启用时，`rope_scaling` 会使用 YaRN 风格参数，包括 `factor=16`、`original_max_position_embeddings=2048`、`beta_fast` 和 `beta_slow`。这用于推理阶段长上下文外推，但当前是否实际训练长上下文，需要结合后续实验结果判断。

## GQA 注意力

`Attention` 中 q/k/v 分别来自：

- `q_proj`: `hidden_size -> num_attention_heads * head_dim`
- `k_proj`: `hidden_size -> num_key_value_heads * head_dim`
- `v_proj`: `hidden_size -> num_key_value_heads * head_dim`
- `o_proj`: `num_attention_heads * head_dim -> hidden_size`

当 `num_key_value_heads < num_attention_heads` 时，就是 GQA。代码通过 `repeat_kv` 把较少的 k/v head 复制到 query head 数量。这样训练和计算形式仍像多头注意力，但 KV-Cache 中存储的 k/v head 更少，推理显存更低。

默认配置是 8 个 query heads、4 个 kv heads，因此每组 kv 被 2 个 query heads 共享。

## Flash Attention 与普通注意力

如果环境支持 `torch.nn.functional.scaled_dot_product_attention` 且满足条件，代码会走 PyTorch SDPA 路径；否则手写 attention scores：

```text
scores = q @ k^T / sqrt(head_dim)
scores += causal_mask
scores += padding_mask
output = softmax(scores) @ v
```

这个设计兼顾可读性和性能：学习时能看到完整 attention 公式，实际运行时又可以利用 PyTorch 的优化 kernel。

## SwiGLU FFN

`FeedForward` 使用 Qwen/LLaMA 常见的 SwiGLU 结构：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

它比普通 `Linear -> activation -> Linear` 多一条门控分支。`gate_proj` 控制哪些通道被激活，`up_proj` 提供候选特征，二者相乘后再投回 hidden size。意义是提升 FFN 表达能力，同时保持结构简单。

## MoE FFN

`MOEFeedForward` 用 gate 给每个 token 计算 expert 分数，并取 top-k expert。默认配置为：

- `num_experts=4`
- `num_experts_per_tok=1`
- `norm_topk_prob=True`
- `router_aux_loss_coef=5e-4`

Top-1 routing 的含义是每个 token 只进入一个 expert，因此 MoE 的总参数量变大，但单 token 激活参数量接近 Dense 模型。代码里 `aux_loss` 用 expert 负载和平均 gate 概率计算，用于降低所有 token 都挤到少数 expert 的 routing collapse 风险。

## Decoder Block

`MiniMindBlock` 是标准 pre-norm 残差结构：

```text
h = h + Attention(RMSNorm(h))
h = h + MLP(RMSNorm(h))
```

如果 `use_moe=False`，MLP 是 Dense FFN；如果 `use_moe=True`，MLP 是 MoE FFN。这样 Dense/MoE 的差异被局部封装在 block 内，外部训练脚本不需要区分。

## Causal LM 与权重绑定

`MiniMindForCausalLM` 继承 `PreTrainedModel` 和 `GenerationMixin`。它内部包含：

- `MiniMindModel`
- `lm_head`
- `model.embed_tokens.weight = self.lm_head.weight`

embedding 和 lm_head 权重绑定可以减少参数量，也让输入 token 表示和输出词表投影共享语义空间。

训练时，如果传入 `labels`，模型会计算 shift 后的 cross entropy，并忽略 `-100` 标签。MoE 的辅助 loss 通过 `MoeCausalLMOutputWithPast(aux_loss=...)` 返回，由训练脚本加到主 loss 上。

## KV-Cache 与 generate

`Attention.forward` 支持 `past_key_value` 和 `use_cache`。生成时只把最新 token 送入模型，并把历史 k/v 拼接到 cache 中，避免每一步重复计算完整上下文。

`generate` 实现了：

- `temperature`
- `top_k`
- `top_p`
- `repetition_penalty`
- `num_return_sequences`
- `eos_token_id` 提前停止
- `streamer`
- `return_kv`

这部分的意义是：模型核心是纯 PyTorch 写的，但推理接口已经具备基础自回归生成能力，可以直接用于 rollout、调试和样例对比。
