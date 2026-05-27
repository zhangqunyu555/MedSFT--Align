# MiniMind/TinyChatLM 深度学习复现学习文档

## 1. 我现在已经完成了什么工作

这次复现的核心工作，是围绕 MiniMind/TinyChatLM 这种轻量级中文小模型，跑通一条从基础语言模型到后训练和对齐的完整链路：

```text
Tokenizer -> Pretrain -> Full SFT -> LoRA SFT -> DPO -> PPO / GRPO
```

你已经做的不只是“跑了几个脚本”，而是把大模型训练里几个关键阶段都拆开验证了一遍：

- 实现并学习了一个 Decoder-Only Transformer 模型结构。
- 使用 MiniMind tokenizer 和 chat template 组织中文对话数据。
- 实现了 Pretrain 数据、SFT 数据、DPO 偏好数据、RLAIF prompt 数据和 Agent 工具数据的 Dataset 层。
- 跑通了 Pretrain、Full SFT、LoRA SFT、DPO、PPO、GRPO 等训练入口。
- 记录了 Pretrain 和 Full SFT 在相同 SFT 验证集上的 PPL 对比。
- 观察并总结了 PPO/GRPO 阶段 reward 偏低、生成过长、策略更新弱等现象。
- 复盘了参数量、路径、依赖、SwanLab、Reward Model、LoRA 路线等实际工程问题。

用一句话概括：

> 你完成的是一个 50M-60M 级中文小模型训练系统复现，而不是单点模型调用。它的价值在于理解每个训练阶段的目标、输入、loss、权重流转、指标和容易踩坑的地方。

当前代码主要包括：

| 模块 | 文件 | 你完成/学习的内容 |
| --- | --- | --- |
| 模型结构 | `model/model_minimind.py` | RMSNorm、RoPE、GQA、SwiGLU、KV-Cache、MoE、Causal LM |
| LoRA | `model/model_lora.py` | 低秩 adapter 注入、冻结、保存、加载、合并 |
| Tokenizer | `trainer/train_tokenizer.py`、`model/tokenizer_config.json` | ByteLevel BPE、特殊 token、chat template |
| Dataset | `dataset/lm_dataset.py` | Pretrain/SFT/DPO/RLAIF/Agent 数据组织与 loss mask |
| 训练工具 | `trainer/trainer_utils.py` | DDP、seed、lr、checkpoint、resume、参数量统计 |
| 预训练 | `trainer/train_pretrain.py` | next-token prediction |
| 全参 SFT | `trainer/train_full_sft.py` | assistant-only 对话微调 |
| LoRA SFT | `trainer/train_lora.py` | 参数高效微调 |
| DPO | `trainer/train_dpo.py` | chosen/rejected 离线偏好优化 |
| PPO | `trainer/train_ppo.py` | actor-critic RLHF |
| GRPO | `trainer/train_grpo.py` | 组内 reward 标准化 RL |
| Rollout | `trainer/rollout_engine.py` | Torch 与 SGLang 推理后端 |
| 蒸馏/Agent | `trainer/train_distillation.py`、`trainer/train_agent.py` | teacher-student、工具调用 RL |

实验上，你已经有一组非常重要的量化结果：

| 模型 | avg_loss | PPL | valid_tokens |
| --- | ---: | ---: | ---: |
| Pretrain | 4.6624 | 105.8949 | 685 |
| Full SFT | 2.6158 | 13.6782 | 685 |

这个结果说明：在 SFT-style 验证集上，Full SFT 明显更适合 user-assistant 对话格式。它是“Pretrain 学续写，SFT 学对话”的量化证据。

需要注意：当前验证集只有 10 examples / 685 valid tokens，所以它适合作为流程验证和初步对比，还不能写成最终严格 benchmark。

## 2. 整体训练流程在做什么

先用大白话讲完整流程。

### 2.1 Tokenizer：把文本变成模型能处理的 token

模型不能直接理解中文字符串，它只能处理整数 id。Tokenizer 的作用就是把：

```text
用户：为什么天空是蓝色的？
助手：这是因为瑞利散射...
```

转换成：

```text
[token_id_1, token_id_2, token_id_3, ...]
```

你的 tokenizer 使用 MiniMind 自带词表和 chat template。它不仅负责切词，还负责把 system/user/assistant/tool 这些结构渲染成统一格式，例如：

```text
<|im_start|>user
为什么天空是蓝色的？<|im_end|>
<|im_start|>assistant
...
<|im_end|>
```

这一步非常重要，因为后面的 SFT、DPO、PPO/GRPO 都依赖相同的对话格式。如果 tokenizer 或 chat template 不一致，模型看到的数据分布就会不一致。

### 2.2 Pretrain：学习通用语言续写

Pretrain 阶段的目标是 next-token prediction：

```text
给定前面的 token，预测下一个 token。
```

它学习的是语言分布和续写能力，不专门学习“用户问，助手答”。所以 pretrain 模型可能会中文，也可能能续写文本，但不一定知道如何稳定地扮演 assistant。

你的 `train_pretrain.py` 默认从随机初始化开始：

```text
from_weight = none
save_weight = pretrain
```

输出权重类似：

```text
pretrain_768.pth
```

### 2.3 Full SFT：学习 user-assistant 对话格式

Full SFT 阶段从 pretrain 权重开始，使用对话数据训练。它的核心目标不是单纯降低普通文本 loss，而是让模型学会：

```text
用户问题 -> 助手回答
```

SFT 的关键是 assistant-only loss：模型可以看到 system/user/assistant 上下文，但只对 assistant 输出部分计算 loss。这样模型不会被训练去生成用户问题，而是专注学习助手回答。

你的 `train_full_sft.py` 默认：

```text
from_weight = pretrain
save_weight = full_sft
```

这也是为什么 Full SFT 在 SFT 验证集上的 PPL 从 105.8949 降到 13.6782。

### 2.4 LoRA SFT：在已有 SFT 模型上做参数高效适配

LoRA 不更新全模型参数，而是在部分线性层旁边加低秩增量：

```text
原输出 = W x
LoRA 输出 = W x + B(Ax)
```

这样只训练 A/B 两个小矩阵，就能适配新任务或新领域。

你的 LoRA 默认从 `full_sft` 开始，而不是从 `pretrain` 开始。这条路线更合理：

```text
pretrain -> full_sft -> lora_sft
```

因为 full_sft 已经学会了对话格式，LoRA 只需要学习任务增量。如果从 pretrain 直接 LoRA，adapter 要同时学对话格式和任务能力，效果更不稳定。

### 2.5 DPO：用 chosen/rejected 做离线偏好对齐

DPO 使用偏好数据：

```text
prompt
chosen response
rejected response
```

它不需要在线生成，也不需要外部 reward model。它做的是：让 policy model 相对于 reference model 更偏向 chosen，而不是 rejected。

你的 DPO 训练中有两个模型：

- policy model：可训练。
- reference model：冻结。

DPO 的核心是比较 chosen/rejected 的 log probability margin。

### 2.6 PPO / GRPO：用 reward model 做在线强化对齐

PPO 和 GRPO 比 DPO 更复杂。它们会让模型先在线生成回答，然后用 reward model 和规则奖励打分，再根据 reward 更新模型。

PPO 有 actor、critic、reference、reward model：

```text
actor 负责生成
critic 估计 value
reference 控制 KL 偏离
reward model 给回答打分
```

GRPO 不需要 critic。它对同一个 prompt 生成多条回答，用组内 reward 均值和方差计算 advantage。

你已经观察到 PPO/GRPO 中 reward 多数为负、平均生成长度容易达到上限、actor loss 接近 0。这说明链路跑通了，但策略质量、停止能力和 reward 设计还需要继续调。

### 2.7 MoE：扩大总容量，但控制激活参数

MoE 把 Dense FFN 替换成多个 expert。每个 token 只路由到部分 expert，例如 Top-1 routing：

```text
总参数很多，但每个 token 只激活一个 expert。
```

你的模型里 MoE 通过 `use_moe=True` 打开，并把 `aux_loss` 加入训练 loss，用来避免所有 token 都挤到同一个 expert。

## 3. 模型代码详细解释

模型核心在 `model/model_minimind.py`。它是一个 Decoder-Only Transformer，也就是 GPT/LLaMA/Qwen 这一类自回归语言模型。

### 3.1 MiniMindConfig：模型配置中心

`MiniMindConfig` 继承 `PretrainedConfig`。这意味着它能和 Hugging Face 的保存、加载、生成接口兼容。

关键参数：

| 参数 | 作用 |
| --- | --- |
| `hidden_size` | 每个 token 的隐藏向量维度，正式实验用 768 |
| `num_hidden_layers` | Transformer block 层数，正式实验用 8 |
| `vocab_size` | 词表大小，默认 6400 |
| `num_attention_heads` | query head 数，默认 8 |
| `num_key_value_heads` | key/value head 数，默认 4 |
| `head_dim` | 每个 head 的维度 |
| `intermediate_size` | FFN 中间层维度 |
| `max_position_embeddings` | RoPE 最大位置长度 |
| `rope_theta` | RoPE 频率 base |
| `inference_rope_scaling` | 是否启用 YaRN 风格长上下文 scaling |
| `use_moe` | 是否使用 MoE FFN |
| `num_experts` | expert 数量，默认 4 |
| `num_experts_per_tok` | 每个 token 激活几个 expert，默认 1 |
| `router_aux_loss_coef` | MoE 负载均衡 loss 权重 |

这个 config 的意义是让 Dense 模型和 MoE 模型共用同一套代码。训练脚本只需要传：

```bash
--hidden_size 768
--num_hidden_layers 8
--use_moe 0
```

或者：

```bash
--use_moe 1
```

就能切换模型结构。

### 3.2 RMSNorm：稳定激活尺度

`RMSNorm` 做的是均方根归一化：

```text
x_norm = x / sqrt(mean(x^2) + eps)
out = weight * x_norm
```

它和 LayerNorm 的区别是：RMSNorm 不减均值，只缩放幅度。LLaMA/Qwen 类模型大量使用 RMSNorm，因为它简单、稳定、计算更轻。

你的 block 里是 pre-norm：

```text
h = h + Attention(RMSNorm(h))
h = h + MLP(RMSNorm(h))
```

pre-norm 的好处是训练更稳定，梯度更容易传递。

### 3.3 RoPE 和 YaRN：给 attention 注入位置信息

Transformer attention 本身不知道 token 的顺序，所以需要位置编码。你的代码使用 RoPE。

`precompute_freqs_cis` 预先计算 cos/sin：

```text
freqs_cos
freqs_sin
```

`apply_rotary_pos_emb` 把旋转位置编码作用到 q/k：

```text
q_embed = q * cos + rotate_half(q) * sin
k_embed = k * cos + rotate_half(k) * sin
```

为什么只作用到 q/k？因为 attention 分数来自：

```text
q @ k^T
```

把位置信息注入 q/k，就能让 attention 分数感知相对位置。

代码还保留了 YaRN 风格 rope scaling：

```text
inference_rope_scaling=True
rope_scaling = {"factor": 16, ...}
```

它的作用是推理时做长上下文外推。但是否真的能提升长上下文能力，需要单独实验验证。

### 3.4 Attention：MHA/GQA 与 KV-Cache

`Attention` 中有四个核心投影：

```text
q_proj
k_proj
v_proj
o_proj
```

默认：

```text
num_attention_heads = 8
num_key_value_heads = 4
```

这就是 GQA。GQA 的含义是 query head 多，key/value head 少。多个 query head 共享一组 key/value。代码通过 `repeat_kv` 把 kv head 扩展到 query head 数量。

为什么 GQA 有意义？因为推理时 KV-Cache 存的是历史 token 的 k/v。kv head 少，cache 就小，推理显存更省。

attention 计算有两条路径：

1. 如果支持 PyTorch SDPA，就走：

```python
F.scaled_dot_product_attention(...)
```

2. 否则手写：

```text
scores = q @ k^T / sqrt(head_dim)
scores += causal_mask
scores += attention_mask
output = softmax(scores) @ v
```

KV-Cache 在 `past_key_value` 里实现。生成时，不需要每一步重算所有历史 token，只需要把新 token 的 k/v 拼到 cache 里。

### 3.5 FeedForward：SwiGLU

`FeedForward` 使用 SwiGLU：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

它比普通 FFN 多一个 gate 分支。可以理解为：

- `up_proj(x)` 生成候选特征。
- `gate_proj(x)` 决定哪些特征通过。
- 两者相乘后再用 `down_proj` 回到 hidden size。

这类结构在 LLaMA/Qwen 模型中很常见，表达能力比普通 `Linear -> GELU -> Linear` 更强。

### 3.6 MOEFeedForward：多个 expert 和负载均衡

`MOEFeedForward` 里有：

```text
gate: hidden_size -> num_experts
experts: 多个 FeedForward
```

每个 token 先经过 gate 得到 expert 概率，然后 `torch.topk` 选出 top-k expert。默认 `num_experts_per_tok=1`，也就是 Top-1 routing。

核心逻辑：

```text
scores = softmax(gate(x))
topk_weight, topk_idx = topk(scores)
token 进入对应 expert
输出按 topk_weight 加权
```

MoE 的问题是 routing collapse：所有 token 都去少数 expert。代码通过 `aux_loss` 缓解：

```text
aux_loss = load * scores.mean
```

训练脚本会把它加到主 loss：

```text
loss = res.loss + res.aux_loss
```

### 3.7 MiniMindBlock：一个 Decoder 层

每层结构是：

```text
hidden_states = hidden_states + self_attn(input_layernorm(hidden_states))
hidden_states = hidden_states + mlp(post_attention_layernorm(hidden_states))
```

这就是标准 Decoder-only block：

- attention 负责 token 间信息交互。
- FFN/MoE 负责每个 token 的非线性变换。
- residual 负责保留原信息。
- RMSNorm 负责稳定训练。

### 3.8 MiniMindForCausalLM：接入 Hugging Face 风格

`MiniMindForCausalLM` 继承：

```python
PreTrainedModel, GenerationMixin
```

这点很关键：模型核心计算是你自己写的 PyTorch，但外部接口兼容 Hugging Face 风格。

它包含：

```text
MiniMindModel
lm_head
```

并做了权重绑定：

```python
self.model.embed_tokens.weight = self.lm_head.weight
```

这样 embedding 和输出词表投影共享参数，减少参数量。

训练时如果传入 `labels`，会计算 causal LM loss：

```text
logits[..., :-1, :]
labels[..., 1:]
cross_entropy(ignore_index=-100)
```

这就是 next-token prediction。

### 3.9 generate：自回归生成

`generate` 支持：

- `temperature`
- `top_p`
- `top_k`
- `repetition_penalty`
- `eos_token_id`
- `use_cache`
- `num_return_sequences`
- `streamer`

每一步做：

```text
取最后一个位置 logits
temperature 缩放
top-k/top-p 过滤
采样或 argmax
拼到 input_ids 后面
更新 past_key_values
```

这套生成逻辑后续被 rollout engine 使用，用于 PPO/GRPO 在线采样。

## 4. Dataset 与 loss mask 详细解释

Dataset 核心在 `dataset/lm_dataset.py`。它决定每个阶段“模型看到什么”和“哪些 token 参与 loss”。

### 4.1 pre_processing_chat：随机加 system prompt

`pre_processing_chat` 会对普通对话以 20% 概率添加 system prompt：

```text
你是 minimind，一个小巧但有用的语言模型。
You are a helpful AI assistant.
...
```

目的：让模型见过更多 system prompt，增强对系统指令的鲁棒性。

如果数据里有 tools，则不做处理，因为工具调用数据结构更严格，不能随便插入 system。

### 4.2 post_processing_chat：随机移除空 think

chat template 可能产生：

```text
<think>

</think>
```

`post_processing_chat` 会以较高概率移除空 think，只保留一部分空 think 样本。这样模型不会被强制训练成每次都输出空 thinking 标签。

### 4.3 PretrainDataset

输入数据字段：

```json
{"text": "..."}
```

构造过程：

```text
tokens = tokenizer(text)
tokens = [bos] + tokens + [eos]
input_ids = tokens + pad
labels = input_ids.clone()
labels[pad] = -100
```

所以 Pretrain 对所有非 pad token 做 next-token prediction。

目标：学习普通文本的语言分布和续写能力。

### 4.4 SFTDataset

SFT 数据是 conversations。它先用 chat template 渲染成完整对话文本，再生成 labels。

核心边界：

```python
bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n')
eos_id = tokenizer(f'{tokenizer.eos_token}\n')
```

`generate_labels` 逻辑：

1. 先把所有 label 设为 `-100`。
2. 扫描 token 序列。
3. 找到 `<|im_start|>assistant\n`。
4. 从 assistant 内容开始，到 `<|im_end|>\n` 为止，设为真实 label。
5. 其他位置保持 `-100`。

这就是 assistant-only loss。

意义：

```text
模型看到 user/system 上下文，但只学习 assistant 应该怎么回答。
```

### 4.5 DPODataset

DPO 数据包含：

```text
chosen: messages list
rejected: messages list
```

Dataset 分别渲染 chosen 和 rejected：

```text
chosen_prompt = apply_chat_template(chosen)
rejected_prompt = apply_chat_template(rejected)
```

再生成 mask：

```text
mask=1 的位置：assistant 回复区间
mask=0 的位置：prompt/pad 等非训练区间
```

最后返回：

```text
x_chosen = chosen_input_ids[:-1]
y_chosen = chosen_input_ids[1:]
mask_chosen = chosen_loss_mask[1:]
x_rejected = rejected_input_ids[:-1]
y_rejected = rejected_input_ids[1:]
mask_rejected = rejected_loss_mask[1:]
```

训练脚本会把 chosen 和 rejected 拼接到一个 batch 里。

### 4.6 RLAIFDataset

PPO/GRPO 不需要 Dataset 提供 labels，因为回答要在线生成。

`RLAIFDataset` 做的是：

```text
取 conversations[:-1]
apply_chat_template(..., add_generation_prompt=True)
返回 prompt
```

也就是把最后的 assistant answer 去掉，让模型自己 rollout。

### 4.7 AgentRLDataset

Agent 数据返回：

```text
messages
tools
gt
```

其中：

- `messages` 是去掉最后答案后的对话。
- `tools` 是 system message 里解析出的工具定义。
- `gt` 是用于 reward 校验的 ground truth。

Agent RL 中，模型会先生成 tool call，脚本执行工具，再把 tool response 回填给模型继续生成。

## 5. 训练脚本代码详细解释

### 5.1 公共训练组件

很多训练脚本结构相似：

```text
init_distributed_mode
setup_seed
MiniMindConfig
init_model
Dataset/DataLoader
optimizer
autocast
forward
loss.backward
gradient clipping
optimizer.step
checkpoint
```

这些在 `trainer/trainer_utils.py` 中统一支持。

#### DDP

`init_distributed_mode` 根据环境变量 `RANK` 判断是否启用分布式。启用后使用 NCCL，并设置当前 GPU。

#### 随机种子

`setup_seed` 固定：

- Python random
- NumPy
- PyTorch CPU
- PyTorch CUDA
- cuDNN deterministic

目的是提升实验可复现性。

#### 学习率

`get_lr` 使用 cosine decay 风格：

```text
lr * (0.1 + 0.45 * (1 + cos(pi * step / total_steps)))
```

它会从初始学习率平滑下降到约 0.1 倍。

#### checkpoint

`lm_checkpoint` 保存两类东西：

- 普通模型权重：用于推理和下游阶段加载。
- resume 状态：包含 model、optimizer、epoch、step、world_size、wandb_id 等。

权重命名与 hidden size、MoE 有关：

```text
<weight>_<hidden_size>.pth
<weight>_<hidden_size>_moe.pth
```

所以如果 `hidden_size` 不一致，就会找错权重或加载失败。

### 5.2 train_pretrain.py

目标：从头训练基础语言模型。

默认关键参数：

| 参数 | 默认值 |
| --- | --- |
| `save_weight` | `pretrain` |
| `from_weight` | `none` |
| `learning_rate` | `5e-4` |
| `batch_size` | 32 |
| `accumulation_steps` | 8 |
| `data_path` | `../dataset/pretrain_t2t_mini.jsonl` |

核心 forward：

```python
res = model(input_ids, labels=labels)
loss = res.loss + res.aux_loss
loss = loss / args.accumulation_steps
```

`res.loss` 是 causal LM loss。`res.aux_loss` 是 MoE 辅助 loss，不开 MoE 时为 0。

输出：`pretrain_768.pth`。

容易踩坑：

- 从根目录跑和从 trainer 目录跑，数据路径不同。
- hidden size 必须和后续 SFT/eval 一致。

### 5.3 train_full_sft.py

目标：让模型学习对话格式和 assistant 回复。

默认关键参数：

| 参数 | 默认值 |
| --- | --- |
| `save_weight` | `full_sft` |
| `from_weight` | `pretrain` |
| `learning_rate` | `1e-5` |
| `batch_size` | 16 |
| `data_path` | `../dataset/sft_t2t_mini.jsonl` |

代码结构和 pretrain 类似，但 Dataset 换成 `SFTDataset`。所以核心差别不在训练 loop，而在 labels：

```text
PretrainDataset: 全文非 pad token 参与 loss
SFTDataset: 只有 assistant 区间参与 loss
```

输出：`full_sft_768.pth`。

### 5.4 train_lora.py

目标：冻结基座模型，只训练 LoRA adapter。

LoRA 结构在 `model/model_lora.py`：

```python
class LoRA(nn.Module):
    A = Linear(in_features, rank)
    B = Linear(rank, out_features)
```

初始化：

```text
A 高斯初始化
B 零初始化
```

为什么 B 要零初始化？因为训练开始时：

```text
B(Ax)=0
```

模型输出和原 full_sft 完全一致，不会一开始就破坏基座能力。

`apply_lora` 会给部分 `nn.Linear` 增加 `lora` 分支，并 monkey-patch forward：

```python
return original_linear(x) + lora(x)
```

然后冻结非 LoRA 参数：

```text
if 'lora' in name: requires_grad=True
else: requires_grad=False
```

保存时只保存 LoRA 权重：

```text
save_lora
```

部署或导出时可以合并：

```text
W = W + B @ A
```

推荐路线：

```text
pretrain -> full_sft -> lora_sft
```

### 5.5 train_dpo.py

目标：用 chosen/rejected 偏好数据做离线对齐。

两个模型：

```text
policy model: 可训练
reference model: 冻结
```

输入来自 `DPODataset`：

```text
x_chosen, y_chosen, mask_chosen
x_rejected, y_rejected, mask_rejected
```

训练脚本拼接：

```python
x = cat([x_chosen, x_rejected])
y = cat([y_chosen, y_rejected])
mask = cat([mask_chosen, mask_rejected])
```

`logits_to_log_probs` 做：

```text
log_softmax(logits)
gather(labels)
```

然后按 mask 求和得到每个 response 的 log probability。

DPO loss：

```text
pi_logratios = chosen_policy_logp - rejected_policy_logp
ref_logratios = chosen_ref_logp - rejected_ref_logp
logits = pi_logratios - ref_logratios
loss = -logsigmoid(beta * logits)
```

直观理解：

```text
如果 policy 比 reference 更偏向 chosen，loss 变小。
如果 policy 仍偏向 rejected，loss 变大。
```

默认学习率很小：

```text
4e-8
```

原因是 DPO 容易让模型偏离 SFT 能力，低学习率可以降低遗忘风险。

### 5.6 train_ppo.py

目标：在线生成回答，用 reward model 评分，再用 PPO 更新策略。

PPO 有四个角色：

| 角色 | 作用 |
| --- | --- |
| actor | 当前策略模型，负责生成 |
| critic | 估计 value |
| reference | 冻结基线，控制 KL |
| reward model | 给回答打分 |

流程：

1. 从 `RLAIFDataset` 取 prompt。
2. rollout engine 生成 response。
3. `calculate_rewards` 用规则和 reward model 打分。
4. actor 计算 old logp。
5. critic 计算 old value。
6. reference 计算 ref logp。
7. 用 GAE 算 advantage。
8. PPO 多轮更新 actor/critic。

PPO 的核心是 clipped objective：

```text
ratio = exp(new_logp - old_logp)
policy_loss = max(-adv * ratio, -adv * clip(ratio))
```

还会加入 reference KL penalty，避免 policy 偏离太远。

你观察到 reward 多为负、生成长度到上限，说明：

- reward model 对回答质量不满意。
- 模型 eos/停止能力不稳定。
- RL 阶段还需要调 max_gen_len、KL、学习率和 reward。

### 5.7 train_grpo.py

目标：不用 critic，也做在线强化对齐。

GRPO 做法：

```text
对每个 prompt 生成 num_generations 条回答
计算每条 reward
组内标准化 reward 得到 advantage
```

公式直观是：

```text
advantage = (reward - group_mean) / (group_std + eps)
```

这样不需要 critic 估计 baseline。

代码支持两种 loss：

```text
loss_type = grpo
loss_type = cispo
```

默认是 `cispo`，会对 ratio 做上界截断，降低异常大更新。

你观察到 actor loss 接近 0，可能原因：

- 学习率太小。
- reward 都偏低且区分度不足。
- 组内 reward 方差不够。
- clipping 或 CISPO 截断导致更新弱。

### 5.8 rollout_engine.py

rollout engine 是 RL 阶段生成回答的抽象层。

`TorchRolloutEngine`：

- 直接调用当前 PyTorch policy model 的 `generate`。
- 简单，不依赖外部服务。
- 吞吐可能较低。

`SGLangRolloutEngine`：

- 通过 HTTP 调 SGLang 服务。
- 可以让推理引擎负责 rollout。
- 支持 `update_weights_from_disk` 动态更新策略权重。

统一返回：

```python
RolloutResult(
    output_ids,
    completion_ids,
    per_token_logps,
    completions
)
```

PPO/GRPO/Agent RL 都可以复用这个接口。

## 6. 我遇到的问题和学到的工程经验

### 6.1 参数量不是严格 64M

你看到过：

```text
25.83M: hidden_size=512, layers=8
54.47M: hidden_size=768, layers=8
```

所以实验记录里不要硬写“严格 64M”，更严谨是：

> 复现 50M-60M 级 MiniMind 小模型训练链路，正式配置为 hidden_size=768、num_hidden_layers=8，实测约 54.47M 参数。

### 6.2 SwanLab 参数名仍叫 wandb

代码里参数叫：

```text
--use_wandb
--wandb_project
```

但实际：

```python
import swanlab as wandb
```

所以理解上要知道：这是 SwanLab，不是 Weights & Biases。

### 6.3 LoRA 默认配置过重

第一次验证 LoRA 链路，不建议直接跑很大 epochs 或过频繁保存。建议：

```bash
--epochs 1
--batch_size 8
--save_interval 100
```

正式实验再加到 3 epochs 左右。

### 6.4 pretrain 直接 LoRA 效果不如 full_sft 后 LoRA

原因很简单：

```text
pretrain 只学续写
full_sft 才学对话
LoRA 适合在已有能力上做增量适配
```

所以效果路线应是：

```text
pretrain -> full_sft -> lora_sft
```

### 6.5 路径和权重一致性很关键

训练脚本通常在 `trainer/` 目录跑：

```text
../dataset/xxx.jsonl
../out/xxx.pth
```

eval 如果在项目根目录跑，则路径变成：

```text
./dataset/xxx.jsonl
./out/xxx.pth
```

每次实验都必须记录：

```text
运行目录
完整命令
hidden_size
num_hidden_layers
from_weight
save_weight
data_path
use_moe
reward_model_path
```

### 6.6 环境依赖问题

Hugging Face 和 SwanLab 依赖冲突时，固定版本：

```bash
pip install "huggingface_hub==0.36.2" "rich==13.7.1"
```

AutoDL 访问 Hugging Face 不通时，使用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download ...
```

`OMP_NUM_THREADS` 报错时：

```bash
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

### 6.7 SwanLab 500 不是训练代码错

如果出现：

```text
api.swanlab.cn too many 500 error responses
```

说明是云端接口问题。先去掉 `--use_wandb` 本地跑通，之后再打开日志。

## 7. 实验结果怎么解释

### 7.1 Pretrain vs Full SFT PPL

你的结果：

| 模型 | avg_loss | PPL | valid_tokens |
| --- | ---: | ---: | ---: |
| Pretrain | 4.6624 | 105.8949 | 685 |
| Full SFT | 2.6158 | 13.6782 | 685 |

解释：

```text
Pretrain PPL 高，不代表不会中文。
它说明 pretrain 不适应 SFT-style 问答格式。
```

Full SFT PPL 大幅下降，说明模型已经学到：

```text
看到 user prompt 后，应该生成 assistant answer。
```

这就是 SFT 的效果。

### 7.2 为什么这个结果能证明 SFT 有效

因为两个模型在相同验证集上评估。唯一主要差异是 Full SFT 经历了对话数据训练。

PPL 从：

```text
105.8949 -> 13.6782
```

说明 full_sft 对 assistant 回答 token 的预测更确定。换句话说，它更知道 SFT 格式下“下一个回答 token 应该是什么”。

### 7.3 loss 不等于生成质量

训练 loss 是 token-level 指标。它衡量的是给定标准答案时，模型预测下一个 token 的能力。

但开放生成还取决于：

- decoding 参数。
- eos 停止能力。
- 重复惩罚。
- 数据质量。
- 模型容量。
- 事实知识。
- 对齐训练质量。

所以 LoRA loss 下降后，如果生成仍然重复或答非所问，并不矛盾。

### 7.4 PPO/GRPO 指标怎么解释

PPO/GRPO 中 reward 多数为负：

```text
Reward Model 认为生成回答质量不高。
```

平均长度达到最大值：

```text
模型没有及时生成 eos，停止能力不稳定，可能在拖长回答。
```

GRPO actor loss 接近 0：

```text
策略更新信号弱，可能是 learning rate 小、reward 方差不足、advantage 太弱或 clipping 抑制更新。
```

这些现象说明 RL 链路跑通了，但不是最终效果。后续应调：

- `max_gen_len`
- learning rate
- KL coefficient
- reward model
- prompt 数据质量
- eos/停止训练

## 8. 面试怎么讲

### Q1: 为什么 Pretrain PPL 高不代表模型不会中文？

因为评估集是 SFT-style 问答格式。Pretrain 学的是普通文本续写，不一定知道 user-assistant 对话结构。PPL 高说明它不适应这个格式，不代表完全没学到中文。

### Q2: 为什么 SFT 后 PPL 大幅下降？

因为 SFT 用 chat template 和 assistant-only loss 训练模型。模型看到用户问题，只对 assistant 回答算 loss，所以它更会预测 assistant 回答 token。

### Q3: assistant-only loss 为什么重要？

如果对 user/system 也算 loss，模型会学习生成用户问题和系统提示，这和推理目标不一致。assistant-only loss 让训练目标和推理目标一致：给定用户输入，只生成助手回答。

### Q4: LoRA 为什么接 full_sft 更合理？

LoRA 参数量很少，适合做增量适配。full_sft 已经学会对话格式，LoRA 只需要学新任务；pretrain 还没学会问答格式，直接 LoRA 难度更大。

### Q5: DPO 为什么不需要 Reward Model？

DPO 直接使用 chosen/rejected 偏好对，通过 policy 和 reference 的 logprob ratio 差异学习偏好。偏好数据本身提供了监督信号，因此不需要显式 reward model。

### Q6: PPO 和 GRPO 区别是什么？

PPO 需要 critic，用 GAE 估计 advantage；GRPO 不需要 critic，而是对同一个 prompt 生成多条回答，用组内 reward 标准化得到 advantage。PPO 更完整但复杂，GRPO 更轻量但依赖多样本 reward 区分度。

### Q7: 为什么 RL 阶段不能只看 loss？

RL loss 的绝对值不直接等于回答质量。必须同时看 reward、KL、response length、clip ratio、样例输出。否则可能出现 loss 正常但 reward 低、KL 发散或生成长度失控。

### Q8: 小模型实验的价值是什么？

小模型不追求大模型级效果，它的价值是把训练链路和算法机制跑通：Tokenizer、Pretrain、SFT、LoRA、DPO、PPO、GRPO、MoE，以及这些阶段对应的数据格式、loss、指标和工程问题。

### Q9: 参数量为什么不是严格 64M？

参数量由 hidden size、层数、词表大小、attention/FFN 结构、是否权重共享、是否 MoE 决定。`hidden_size=768, layers=8` 实测约 54.47M，所以更严谨表述是 50M-60M 级小模型。

### Q10: PPO/GRPO reward 低、长度到上限说明什么？

说明模型生成质量还不稳定，reward model 给分低，而且模型可能没有学好停止。需要调 max generation length、eos 数据、KL、学习率和 reward model。

## 9. 总结

这次 MiniMind/TinyChatLM 复现可以这样总结：

> 我复现了一个 50M-60M 级中文小模型训练链路，覆盖 Tokenizer、Pretrain、Full SFT、LoRA SFT、DPO、PPO 和 GRPO。模型结构上实现并理解了 RMSNorm、RoPE、GQA、SwiGLU、KV-Cache、MoE 等模块；数据层实现了 Pretrain、SFT、DPO、RLAIF 和 Agent RL 的 Dataset 组织与 loss mask；训练层跑通了从 next-token prediction 到 assistant-only SFT、低秩适配、离线偏好优化和在线强化对齐的完整流程。实验中，Full SFT 在相同 SFT 验证集上的 PPL 从 Pretrain 的 105.89 降到 13.68，说明 SFT 显著提升了模型对 user-assistant 对话格式的建模能力。同时，PPO/GRPO 阶段暴露出 reward 偏低、生成过长和策略更新较弱等问题，这让我进一步理解了小模型 RL 对齐中 reward、KL、生成长度和学习率控制的重要性。

最核心的理解是：

> Pretrain 让模型会续写，SFT 让模型会对话，LoRA 让模型低成本适配任务，DPO 让模型学习离线偏好，PPO/GRPO 让模型通过在线 reward 调整行为。小模型效果不一定强，但它非常适合用来理解大模型训练系统的完整工程链路。
