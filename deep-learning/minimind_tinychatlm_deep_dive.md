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

这一章专门按“代码怎么跑”的顺序讲。每个训练脚本本质上都在回答同一组问题：

```text
batch 从哪里来？
batch 里每个字段是什么？
哪些张量搬到 GPU？
模型 forward 返回什么？
loss 怎么构造？
什么时候 backward？
什么时候 optimizer.step？
checkpoint 保存了什么？
```

如果能把这几个问题讲清楚，训练方法就不是抽象概念，而是能真正 debug 的代码路径。

### 5.1 训练脚本通用执行骨架

Pretrain、Full SFT、LoRA、DPO、PPO、GRPO 都遵循类似的主流程：

```text
argparse
init_distributed_mode
setup_seed
MiniMindConfig
lm_checkpoint(load resume)
autocast / GradScaler
init_model
Dataset / DataLoader
optimizer / scheduler
train_epoch
save checkpoint
destroy_process_group
```

#### 5.1.1 `argparse`：所有实验配置都从命令行进来

每个训练脚本都有：

```python
parser = argparse.ArgumentParser(...)
parser.add_argument("--save_dir", ...)
parser.add_argument("--epochs", ...)
parser.add_argument("--batch_size", ...)
parser.add_argument("--learning_rate", ...)
parser.add_argument("--hidden_size", ...)
parser.add_argument("--num_hidden_layers", ...)
parser.add_argument("--use_moe", ...)
parser.add_argument("--data_path", ...)
parser.add_argument("--from_weight", ...)
args = parser.parse_args()
```

这意味着一次实验是否可复现，核心取决于命令行参数是否完整记录。尤其是：

```text
hidden_size
num_hidden_layers
from_weight
save_weight
data_path
use_moe
dtype
accumulation_steps
```

只要这些不一致，就可能出现参数量不一致、权重找不到、权重 shape 不匹配、loss 不可比等问题。

#### 5.1.2 `init_distributed_mode()`：判断是否使用 DDP

代码逻辑是：

```python
if int(os.environ.get("RANK", -1)) == -1:
    return 0

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
return local_rank
```

解释：

- 如果环境变量里没有 `RANK`，说明不是 `torchrun` 启动，直接单卡/单进程训练。
- 如果有 `RANK`，就初始化 NCCL 分布式进程组。
- `LOCAL_RANK` 决定当前进程绑定哪张 GPU。

训练脚本后面会根据它改写：

```python
if dist.is_initialized():
    args.device = f"cuda:{local_rank}"
```

这就是为什么 DDP 启动时不能手动所有进程都用 `cuda:0`。

#### 5.1.3 `setup_seed()`：固定随机性

`setup_seed` 同时固定：

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

作用：让数据 shuffle、参数初始化、dropout 等尽量可复现。

注意：分布式训练里通常用：

```python
setup_seed(42 + dist.get_rank())
```

每个 rank 的 seed 稍有不同，避免所有卡采样完全一样。

#### 5.1.4 `MiniMindConfig(...)`：决定模型结构

训练脚本会创建：

```python
lm_config = MiniMindConfig(
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_hidden_layers,
    use_moe=bool(args.use_moe)
)
```

这个对象决定：

- 模型维度。
- 层数。
- attention head 数。
- GQA kv head 数。
- 是否使用 MoE。
- MoE expert 数和 aux loss 系数。

所以 `hidden_size=512` 和 `hidden_size=768` 是两个不同结构的模型，不能混用权重。

#### 5.1.5 `init_model()`：加载 tokenizer、构造模型、加载权重

`init_model` 做三件事：

```python
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = MiniMindForCausalLM(lm_config)
```

如果 `from_weight != 'none'`：

```python
weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
weights = torch.load(weight_path, map_location=device)
model.load_state_dict(weights, strict=False)
```

解释：

- `from_weight='none'`：从随机初始化开始，比如 pretrain。
- `from_weight='pretrain'`：加载 `pretrain_768.pth`，比如 full_sft。
- `from_weight='full_sft'`：加载 `full_sft_768.pth`，比如 LoRA/DPO/PPO/GRPO。
- `moe_suffix='_moe'`：如果 `use_moe=True`，权重名会变成 `xxx_768_moe.pth`。

这就是为什么路径、权重名前缀、hidden size、MoE 开关必须一致。

#### 5.1.6 `autocast_ctx` 和 `GradScaler`：混合精度

代码会根据设备和 dtype 设置：

```python
device_type = "cuda" if "cuda" in args.device else "cpu"
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
```

解释：

- bf16 用 autocast，但通常不需要 GradScaler。
- fp16 动态范围更窄，需要 GradScaler 防止梯度下溢。
- CPU 不使用 CUDA autocast，所以走 `nullcontext()`。

训练时一般是：

```python
with autocast_ctx:
    res = model(...)
    loss = ...
scaler.scale(loss).backward()
```

#### 5.1.7 `SkipBatchSampler`：resume 时跳过已训练 batch

如果中断训练，resume 文件里会记录：

```text
epoch
step
optimizer
scaler
```

恢复时，代码用：

```python
skip = start_step if (epoch == start_epoch and start_step > 0) else 0
batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
```

`SkipBatchSampler` 会在当前 epoch 跳过前 `skip_batches` 个 batch，从中断位置继续训练。

#### 5.1.8 `lm_checkpoint()`：保存推理权重和 resume 状态

保存模型时有两类文件：

```text
../out/<weight>_<hidden_size>.pth
../checkpoints/<weight>_<hidden_size>_resume.pth
```

普通权重只保存模型参数，方便下游加载。resume 权重还保存：

```python
{
    "model": state_dict,
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "step": step,
    "world_size": ...,
    "wandb_id": ...
}
```

PPO 还会额外保存 critic、scheduler 等状态。

### 5.2 Pretrain 和 Full SFT：`train_epoch` 逐行解释

`train_pretrain.py::train_epoch` 和 `train_full_sft.py::train_epoch` 几乎一样，核心差别是 Dataset：

```text
PretrainDataset -> labels 覆盖全文非 pad token
SFTDataset -> labels 只覆盖 assistant 区间
```

也就是说：训练 loop 相同，但任务语义由 labels 决定。

#### 5.2.1 batch 输入

DataLoader 每次返回：

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
```

张量形状通常是：

```text
input_ids: [B, T]
labels:    [B, T]
```

其中：

- `B` 是 batch size。
- `T` 是 max sequence length。
- `input_ids` 是模型输入。
- `labels` 是训练目标，不参与 loss 的位置为 `-100`。

#### 5.2.2 搬到 GPU

```python
input_ids = input_ids.to(args.device)
labels = labels.to(args.device)
```

DataLoader 读出来的张量默认在 CPU。模型在 GPU 上，所以 batch 也必须搬到同一个 device。

如果忘记这一步，会报 device mismatch：

```text
Expected all tensors to be on the same device
```

#### 5.2.3 动态调整学习率

```python
lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

解释：

- `epoch * iters + step` 是全局 step。
- `args.epochs * iters` 是总 step。
- `get_lr` 返回 cosine decay 后的学习率。
- 遍历 `param_groups` 是因为 optimizer 可能有多个参数组。

#### 5.2.4 forward 和 loss

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps
```

`model(input_ids, labels=labels)` 会进入 `MiniMindForCausalLM.forward`：

```python
hidden_states, past_key_values, aux_loss = self.model(...)
logits = self.lm_head(hidden_states)
loss = cross_entropy(logits[..., :-1, :], labels[..., 1:], ignore_index=-100)
```

关键点：

- `res.loss` 是 next-token prediction CE loss。
- `ignore_index=-100` 会忽略 Dataset mask 掉的位置。
- `res.aux_loss` 是 MoE 辅助 loss，Dense 模型时为 0。
- 除以 `accumulation_steps` 是为了梯度累积时保持总梯度尺度不变。

#### 5.2.5 backward

```python
scaler.scale(loss).backward()
```

如果 dtype 是 fp16，GradScaler 会把 loss 放大再反传，避免梯度太小下溢。如果是 bf16，scaler disabled，本质上等价普通 backward。

#### 5.2.6 梯度累积和 optimizer step

```python
if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

解释：

- 每个 micro-batch 都 backward。
- 每 `accumulation_steps` 个 micro-batch 才更新一次参数。
- `unscale_` 后才能正确做梯度裁剪。
- `clip_grad_norm_` 防止梯度爆炸。
- `zero_grad(set_to_none=True)` 更省显存。

如果 epoch 结束时还有不足一个 accumulation 的残留 batch，代码会在循环后再补一次 step。

#### 5.2.7 日志

```python
current_loss = loss.item() * args.accumulation_steps
current_aux_loss = res.aux_loss.item()
current_logits_loss = current_loss - current_aux_loss
```

因为训练时 loss 除过 `accumulation_steps`，日志里乘回来，显示真实 batch loss。

记录指标：

```text
loss
logits_loss
aux_loss
learning_rate
epoch_time
```

#### 5.2.8 checkpoint

```python
if (step % args.save_interval == 0 or step == iters) and is_main_process():
    model.eval()
    state_dict = raw_model.state_dict()
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
    lm_checkpoint(...)
    model.train()
```

解释：

- 只在 main process 保存，避免多卡同时写文件。
- 保存前 `model.eval()`，保存后切回 `model.train()`。
- 权重转 half/cpu，减少磁盘占用。
- `lm_checkpoint` 额外保存 resume 状态。

#### 5.2.9 Pretrain 和 SFT 到底差在哪

Pretrain 的 Dataset：

```text
labels = input_ids.clone()
labels[pad] = -100
```

所以几乎全文都参与 loss。

SFT 的 Dataset：

```text
labels 默认全是 -100
只把 assistant 区间设为真实 token id
```

所以训练 loop 虽然一样，但优化目标完全不同：

```text
Pretrain: 学文本续写
SFT: 学看到用户问题后如何生成 assistant 回答
```

### 5.3 LoRA SFT：从 adapter 到训练 loop

LoRA 相关代码在 `model/model_lora.py` 和 `trainer/train_lora.py`。

#### 5.3.1 `LoRA.forward()` 为什么是 `B(A(x))`

代码结构：

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.B(self.A(x))
```

原始线性层是：

```text
y = W x
```

LoRA 加的是低秩增量：

```text
y = W x + B(Ax)
```

其中：

- `A` 把高维输入压到 rank 维。
- `B` 再把 rank 维投回输出维。
- `B @ A` 的矩阵秩最多是 rank。

所以 LoRA 参数量是：

```text
in_features * rank + rank * out_features
```

远小于完整 `in_features * out_features`。

#### 5.3.2 A 高斯初始化，B 零初始化

代码里：

```python
self.A.weight.data.normal_(mean=0.0, std=0.02)
self.B.weight.data.zero_()
```

因为 B 是 0，所以训练一开始：

```text
B(Ax) = 0
```

这意味着刚注入 LoRA 时模型输出不变。这样不会破坏 `full_sft` 已有能力。

#### 5.3.3 `apply_lora()` 如何 monkey-patch forward

核心代码逻辑：

```python
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
        lora = LoRA(...).to(model.device)
        setattr(module, "lora", lora)
        original_forward = module.forward

        def forward_with_lora(x, layer1=original_forward, layer2=lora):
            return layer1(x) + layer2(x)

        module.forward = forward_with_lora
```

逐句解释：

- `model.named_modules()` 遍历模型中所有子模块。
- `isinstance(module, nn.Linear)` 只处理线性层。
- `module.weight.shape[0] == module.weight.shape[1]` 表示只给方阵线性层加 LoRA。
- `setattr(module, "lora", lora)` 把 LoRA 模块挂到原 Linear 上。
- `original_forward = module.forward` 保存原来的线性层计算。
- 新的 `forward_with_lora` 返回原输出加 LoRA 输出。
- `module.forward = forward_with_lora` 直接替换模块 forward。

这叫 monkey-patch。好处是实现简单；缺点是和 `torch.compile` 不兼容，所以脚本里会自动关闭 compile。

#### 5.3.4 只训练 LoRA 参数

`train_lora.py` 里：

```python
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False
```

然后 optimizer 只拿 LoRA 参数：

```python
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

所以 backward 时虽然整个模型参与 forward，但只有 LoRA 参数会产生可更新梯度。

训练 step 里裁剪的也是：

```python
clip_grad_norm_(lora_params, args.grad_clip)
```

这就是真正的参数高效微调。

#### 5.3.5 `save_lora()` 只保存 adapter

核心逻辑：

```python
for name, module in raw_model.named_modules():
    if hasattr(module, 'lora'):
        lora_state = {
            f'{clean_name}.lora.{k}': v.cpu().half()
            for k, v in module.lora.state_dict().items()
        }
        state_dict.update(lora_state)
torch.save(state_dict, path)
```

只保存 `.lora.` 里的 A/B 参数，不保存完整模型，所以文件小。

#### 5.3.6 `load_lora()` 如何加载

`load_lora` 会从 checkpoint 中筛选当前模块对应的 key：

```python
lora_state = {
    k.replace(f'{name}.lora.', ''): v
    for k, v in state_dict.items()
    if f'{name}.lora.' in k
}
module.lora.load_state_dict(lora_state)
```

前提：模型已经先执行过 `apply_lora()`，否则模块上没有 `module.lora`。

#### 5.3.7 `merge_lora()` 如何合并

核心：

```python
state_dict[f'{name}.weight'] = module.weight.data.clone()
state_dict[f'{name}.weight'] += module.lora.B.weight.data @ module.lora.A.weight.data
```

因为 LoRA 增量就是：

```text
delta_W = B @ A
```

合并后，推理时不需要额外 lora 分支：

```text
W_merged = W + delta_W
```

### 5.4 DPO：从 logprob 到偏好 loss

DPO 代码主要在 `trainer/train_dpo.py`。

#### 5.4.1 `logits_to_log_probs(logits, labels)`

输入：

```text
logits: [B, T, V]
labels: [B, T]
```

其中：

- `B` 是 batch size。
- `T` 是序列长度。
- `V` 是 vocab size。

代码：

```python
log_probs = F.log_softmax(logits, dim=2)
log_probs_per_token = torch.gather(
    log_probs,
    dim=2,
    index=labels.unsqueeze(2)
).squeeze(-1)
```

解释：

- `log_softmax(logits, dim=2)` 把每个位置的 vocab logits 变成 log probability。
- `labels.unsqueeze(2)` 让 labels 从 `[B, T]` 变成 `[B, T, 1]`。
- `gather(dim=2, index=...)` 取出目标 token 对应的 log probability。
- 输出是 `[B, T]`，每个位置一个 logp。

#### 5.4.2 `dpo_loss(ref_log_probs, policy_log_probs, mask, beta)`

第一步：只保留 assistant answer 部分：

```python
ref_log_probs = (ref_log_probs * mask).sum(dim=1)
policy_log_probs = (policy_log_probs * mask).sum(dim=1)
```

这里：

```text
mask: [B, T]
```

mask 为 1 的位置是回答 token，mask 为 0 的位置是 prompt/pad。

第二步：把 batch 分成 chosen 和 rejected：

```python
batch_size = ref_log_probs.shape[0]
chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
reject_ref_log_probs = ref_log_probs[batch_size // 2:]
chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
reject_policy_log_probs = policy_log_probs[batch_size // 2:]
```

这是因为 `train_epoch` 里拼 batch 时写的是：

```python
x = torch.cat([x_chosen, x_rejected], dim=0)
```

所以前半是 chosen，后半是 rejected。

第三步：计算 policy 和 reference 的偏好 margin：

```python
pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
logits = pi_logratios - ref_logratios
```

含义：

- `pi_logratios`：policy 对 chosen 相比 rejected 的偏好程度。
- `ref_logratios`：reference 对 chosen 相比 rejected 的偏好程度。
- `logits`：policy 比 reference 多出来的偏好提升。

第四步：DPO loss：

```python
loss = -F.logsigmoid(beta * logits)
return loss.mean()
```

如果 policy 已经比 reference 更偏 chosen，`logits` 大，`logsigmoid` 接近 0，loss 小。

#### 5.4.3 `train_dpo.py::train_epoch`

batch 字段：

```python
x_chosen = batch['x_chosen'].to(args.device)
x_rejected = batch['x_rejected'].to(args.device)
y_chosen = batch['y_chosen'].to(args.device)
y_rejected = batch['y_rejected'].to(args.device)
mask_chosen = batch['mask_chosen'].to(args.device)
mask_rejected = batch['mask_rejected'].to(args.device)
```

拼接：

```python
x = torch.cat([x_chosen, x_rejected], dim=0)
y = torch.cat([y_chosen, y_rejected], dim=0)
mask = torch.cat([mask_chosen, mask_rejected], dim=0)
```

reference forward：

```python
with torch.no_grad():
    ref_outputs = ref_model(x)
    ref_logits = ref_outputs.logits
ref_log_probs = logits_to_log_probs(ref_logits, y)
```

为什么 `torch.no_grad()`？reference model 是冻结基线，不需要梯度，省显存，也防止被更新。

policy forward：

```python
outputs = model(x)
logits = outputs.logits
policy_log_probs = logits_to_log_probs(logits, y)
```

policy 是要训练的，所以不能 no_grad。

loss：

```python
dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
loss = dpo_loss_val + outputs.aux_loss
loss = loss / args.accumulation_steps
```

如果是 MoE，`outputs.aux_loss` 保持 expert 负载均衡；Dense 模型时基本为 0。

后面的 backward、clip、step、save 和 Pretrain/SFT 一样。

### 5.5 PPO：actor-critic 的在线强化对齐

PPO 代码在 `trainer/train_ppo.py`，核心函数是 `ppo_train_epoch`。

#### 5.5.1 `CriticModel`

代码：

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)
        return values
```

解释：

- Critic 复用 MiniMind backbone。
- 不用 `lm_head` 预测 token，而是用 `value_head` 输出每个 token 的 value。
- 输出形状是 `[B, T]`，表示每个位置的价值估计。

#### 5.5.2 `calculate_rewards()`

PPO reward 由规则奖励和 reward model 组成。

长度奖励：

```python
rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
```

thinking 格式奖励：

```python
if '</think>' in response:
    thinking_content, answer_content = response.split('</think>', 1)
    rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
    rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
```

重复惩罚：

```python
rewards[i] -= rep_penalty(answer)
```

reward model 分数：

```python
score = reward_model.get_score(messages, answer)
rewards += reward_model_scores
```

所以 PPO 优化目标不只是“像参考答案”，而是综合长度、格式、重复度和外部 RM 打分。

#### 5.5.3 `ppo_train_epoch` 第一步：prompt tokenize 和 rollout

输入 batch：

```python
prompts = batch["prompt"]
```

tokenize：

```python
enc = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=args.max_seq_len,
    padding_side="left"
).to(args.device)
prompt_length = enc.input_ids.shape[1]
```

这里 `prompt_length=P`。因为 left padding，batch 内 prompt 会 pad 到同一长度。

rollout：

```python
rollout_result = rollout_engine.rollout(
    prompt_ids=enc.input_ids,
    attention_mask=enc.attention_mask,
    num_generations=1,
    max_new_tokens=args.max_gen_len,
    temperature=0.8,
)
gen_out = rollout_result.output_ids
responses_text = rollout_result.completions
```

张量形状：

```text
gen_out: [B, P+R]
```

其中 `P` 是 prompt 长度，`R` 是生成长度。

#### 5.5.4 labels、mask 和 response 区间

代码：

```python
full_mask = (gen_out != tokenizer.pad_token_id).long()
labels = gen_out[:, 1:].clone()
seq_len, resp_start = gen_out.size(1) - 1, prompt_length - 1
```

解释：

- `labels` 是 next-token prediction 的目标，长度是 `P+R-1`。
- `resp_start = prompt_length - 1`，因为 logits 的第 `prompt_length-1` 个位置预测第一个 response token。

构造 response mask：

```python
resp_mask = torch.arange(seq_len).unsqueeze(0) >= resp_start
final_mask = resp_mask & (~labels.eq(pad_id))
```

后面又根据 eos 重新算 response 有效长度：

```python
resp_labels = labels[:, resp_start:]
resp_pad_mask = ~resp_labels.eq(pad_id)
eos_mask = resp_labels.eq(eos_id) & resp_pad_mask
resp_lengths = eos_pos + 1 if has_eos else pad_mask.sum
resp_policy_mask = (resp_idx < resp_lengths) & resp_pad_mask
```

形状：

```text
resp_labels:      [B, R]
resp_policy_mask: [B, R]
resp_lengths:     [B]
```

`resp_policy_mask` 决定 PPO loss 只算 response 的有效 token，eos 后和 pad 不参与训练。

#### 5.5.5 old value、old logp、reference logp

这些都在 `torch.no_grad()` 下算，因为 rollout 阶段只是收集旧策略数据。

critic old value：

```python
values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
old_resp_values = values_seq[:, resp_start:-1] * resp_value_mask
```

形状：

```text
values_seq:       [B, P+R]
old_resp_values:  [B, R]
```

actor old logp：

```python
logits = actor_for_rollout(input_ids=gen_out, attention_mask=full_mask).logits
old_resp_logp = log_softmax(logits[:, :-1]).gather(labels)[:, resp_start:]
```

形状：

```text
old_resp_logp: [B, R]
```

reference logp：

```python
ref_logp_all = log_softmax(ref_model(...).logits[:, :-1]).gather(labels)
ref_resp_logp = ref_logp_all[:, resp_start:]
```

reference 用来计算 KL penalty，控制 policy 不要偏离 SFT 太远。

#### 5.5.6 token reward 和 GAE

外部 reward 是 response 级别的：

```text
rewards: [B]
```

代码把它加到每条 response 的最后有效 token：

```python
token_rewards = torch.zeros_like(old_resp_logp)
last_idx = resp_lengths - 1
token_rewards[torch.arange(B), last_idx] += rewards
```

然后反向递推 GAE：

```python
lastgaelam = 0
for t in reversed(range(gen_len)):
    nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
    delta = token_rewards[:, t] + gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + gamma * lam * lastgaelam
    advs_rev.append(lastgaelam)
advantages = stack(reverse(advs_rev))
returns = advantages + old_resp_values
```

含义：

- `advantages`: 当前 token 的动作比 critic 预期好多少。
- `returns`: value head 应该回归的目标。

然后标准化 advantage：

```python
adv_mean = masked_mean(advantages)
adv_var = masked_var(advantages)
advantages = (advantages - adv_mean) / sqrt(adv_var + 1e-8)
```

这能让 PPO 更新更稳定。

#### 5.5.7 PPO minibatch 更新

PPO 对同一批 rollout 做多轮更新：

```python
for ppo_epoch in range(args.ppo_update_iters):
    b_inds = torch.randperm(B)
    for i in range(0, B, mb_size):
        inds = b_inds[i:i + mb_size]
```

重新计算当前 policy logp：

```python
res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
mb_logp_all = log_softmax(res.logits[:, :-1]).gather(labels[inds])
mb_resp_logp = mb_logp_all[:, resp_start:]
```

计算 ratio：

```python
log_ratio = mb_resp_logp - old_resp_logp[inds]
ratio = torch.exp(log_ratio)
```

`ratio` 表示新策略相对旧策略在同一个 token 上的概率变化。

approx KL：

```python
approx_kl = 0.5 * (log_ratio ** 2)
```

如果 KL 太大：

```python
if approx_kl_val > args.early_stop_kl:
    stop_ppo = True
```

这避免 policy 一步偏离太远。

clip fraction：

```python
clipfrac = ((ratio - 1).abs() > clip_epsilon).mean()
```

它表示有多少 token 的 ratio 超过裁剪范围。

reference KL penalty：

```python
kl_ref_penalty = exp(ref_logp - mb_logp) - (ref_logp - mb_logp) - 1
```

这是一个非负形式的 KL 近似，用来约束 policy 不要离 reference 太远。

policy loss：

```python
policy_loss = max(
    -adv * ratio,
    -adv * clamp(ratio, 1-eps, 1+eps)
) + kl_coef * kl_ref_penalty
```

value loss：

```python
value_loss = 0.5 * max(
    (new_value - returns)^2,
    (clamp(new_value, old_value - clip, old_value + clip) - returns)^2
)
```

总 loss：

```python
loss = policy_loss + vf_coef * value_loss + aux_loss
```

如果 early stop，为了避免 DDP 通信死锁，代码不是直接 break，而是让 loss 乘 0，保证 forward-backward 闭环还存在。

#### 5.5.8 actor/critic 更新

```python
loss.backward()
clip_grad_norm_(actor_model.parameters(), args.grad_clip)
clip_grad_norm_(critic_model.parameters(), args.grad_clip)
actor_optimizer.step()
critic_optimizer.step()
actor_scheduler.step()
critic_scheduler.step()
actor_optimizer.zero_grad()
critic_optimizer.zero_grad()
```

PPO 同时更新 actor 和 critic：

- actor 学“怎么生成更高 reward 的 token”。
- critic 学“当前状态的 value 应该是多少”。

#### 5.5.9 PPO checkpoint

保存时：

```python
torch.save(actor_state, ppo_actor_768.pth)
lm_checkpoint(..., critic_model=critic_model, critic_optimizer=critic_optimizer, ...)
```

普通推理主要用 actor 权重。resume 需要同时恢复 critic、两个 optimizer、两个 scheduler。

### 5.6 GRPO：无 critic 的组内相对优化

GRPO 代码在 `trainer/train_grpo.py`，核心函数是 `grpo_train_epoch`。

#### 5.6.1 `calculate_rewards()` 如何处理多 generation

GRPO 对每个 prompt 生成 `num_generations` 条回答。假设：

```text
B = prompt 数量
G = num_generations
```

responses 的长度是：

```text
B * G
```

代码通过：

```python
for i in range(batch_size):
    for j in range(args.num_generations):
        response_idx = i * args.num_generations + j
```

找到第 i 个 prompt 的第 j 个回答。

每条回答同样计算：

- 长度奖励。
- thinking 标签奖励。
- 重复惩罚。
- reward model 分数。

输出：

```text
rewards: [B * G]
```

#### 5.6.2 prompt tokenize 和左截断

```python
prompt_inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    return_token_type_ids=False,
    padding_side="left",
    add_special_tokens=False
).to(args.device)
```

如果超过最大 prompt 长度：

```python
prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]
```

左截断保留最后的上下文，因为最近的 user query 通常最重要。

#### 5.6.3 rollout 一次生成 `B * G` 条 completion

```python
rollout_result = rollout_engine.rollout(
    prompt_ids=prompt_inputs["input_ids"],
    attention_mask=prompt_inputs["attention_mask"],
    num_generations=args.num_generations,
    max_new_tokens=args.max_gen_len,
    temperature=0.8,
)
```

返回：

```text
outputs:            [B*G, P+R]
completion_ids:     [B*G, R]
completions:        list[str], length B*G
old_per_token_logps:[B*G, R]
```

`old_per_token_logps` 是 rollout 时旧策略生成这些 token 的 logprob，是计算 ratio 的基准。

#### 5.6.4 当前 policy logp 和 reference logp

如果使用 SGLang 或 MoE，代码重新 forward 当前模型：

```python
res = model_unwrapped(outputs)
logits = res.logits[:, :-1, :]
per_token_logps = log_softmax(logits).gather(outputs[:, 1:])
per_token_logps = per_token_logps[:, -completion_ids.size(1):]
```

只取 completion 的后 R 个 token logp。

reference logp：

```python
ref_per_token_logps = compute_per_token_logps(ref_model, outputs, completion_ids.size(1))
```

形状都是：

```text
[B*G, R]
```

#### 5.6.5 组内 reward 标准化

```python
grouped_rewards = rewards.view(-1, args.num_generations)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

形状：

```text
grouped_rewards: [B, G]
advantages:      [B*G]
```

这就是 GRPO 不需要 critic 的原因：同一个 prompt 下多条回答互相比较，组内均值就是 baseline。

#### 5.6.6 `completion_mask` 排除 eos 后 token

```python
is_eos = completion_ids == tokenizer.eos_token_id
eos_idx = ...
completion_mask = arange(R) <= eos_idx
```

如果某条回答提前出现 eos，eos 后面的 pad 或无效 token 不参与 loss。

形状：

```text
completion_mask: [B*G, R]
```

#### 5.6.7 KL 和 ratio

```python
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1
ratio = torch.exp(per_token_logps - old_per_token_logps)
```

解释：

- `per_token_kl` 控制当前 policy 不要离 reference 太远。
- `ratio` 比较当前 policy 和 rollout 旧 policy 对同一 token 的概率变化。

#### 5.6.8 GRPO loss 和 CISPO loss

GRPO 分支：

```python
clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
per_token_loss1 = ratio * advantages
per_token_loss2 = clipped_ratio * advantages
per_token_loss = -(min(loss1, loss2) - beta * per_token_kl)
```

这是 PPO-style clip。

CISPO 分支：

```python
clamped_ratio = torch.clamp(ratio, max=epsilon_high).detach()
per_token_loss = -(clamped_ratio * advantages * per_token_logps - beta * per_token_kl)
```

区别：

- GRPO 用 ratio 和 clipped ratio 取 min。
- CISPO 对 ratio 设上界，并 detach，减少异常大 ratio 带来的不稳定。

最后：

```python
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
loss = (policy_loss + aux_loss) / accumulation_steps
```

#### 5.6.9 更新与 rollout policy 同步

```python
loss.backward()
if step % accumulation_steps == 0:
    clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    rollout_engine.update_policy(model)
```

如果 rollout engine 是 SGLang，`update_policy` 会把新权重保存到共享路径，并通知 SGLang 服务重新加载。

#### 5.6.10 actor loss 接近 0 怎么排查

从代码变量看：

- `rewards` 是否都差不多，导致 `advantages` 很小。
- `std_r` 是否过小，组内没有区分度。
- `learning_rate` 是否太低。
- `ratio` 是否接近 1，说明更新弱。
- `completion_mask.sum()` 是否很小。
- `beta * per_token_kl` 是否压过 policy reward 项。
- `old_per_token_logps` 是否和当前 `per_token_logps` 计算口径一致。

### 5.7 Rollout Engine：训练和生成之间的接口

RL 阶段最容易乱的地方，是“谁负责生成”。你的代码用 `rollout_engine.py` 抽象出来。

#### 5.7.1 `compute_per_token_logps()`

输入：

```text
model
input_ids: [B, T]
n_keep: 只保留最后多少个 token
```

核心：

```python
logits = model(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
ids_row = input_ids[:, -n_keep:]
gather(log_softmax(logits), ids_row)
```

为什么只保留最后 `n_keep` 个？因为 RL loss 只优化 completion，不优化 prompt。

输出：

```text
per_token_logps: [B, n_keep]
```

#### 5.7.2 `TorchRolloutEngine.rollout()`

核心：

```python
output_ids = model.generate(
    input_ids=prompt_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=temperature,
    num_return_sequences=num_generations,
)
completion_ids = output_ids[:, prompt_len:]
per_token_logps = compute_per_token_logps(model, output_ids, completion_len)
completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
```

它完全在 PyTorch 进程里生成，简单可靠，但大规模 rollout 速度可能慢。

#### 5.7.3 `SGLangRolloutEngine.rollout()`

SGLang 分支会把 prompt ids 转成 list，通过 HTTP 请求：

```python
POST /generate
payload = {
    "input_ids": all_input_ids,
    "sampling_params": {...},
    "return_logprob": True,
}
```

返回里取：

```text
meta_info.output_ids
meta_info.output_token_logprobs
```

再 pad 成 tensor：

```text
output_ids
completion_ids
per_token_logps
```

这让 rollout 可以交给推理服务做。

#### 5.7.4 `SGLangRolloutEngine.update_policy()`

核心流程：

```python
unwrapped = model.module if DDP else model
state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
tokenizer.save_pretrained(abs_path)
POST /update_weights_from_disk {"model_path": abs_path}
```

作用：训练进程更新了 policy 后，把新权重同步给 SGLang 服务。

### 5.8 Distillation：teacher soft label 的 KL

蒸馏代码在 `trainer/train_distillation.py`。

#### 5.8.1 `distillation_loss()`

```python
teacher_probs = softmax(teacher_logits / temperature)
student_log_probs = log_softmax(student_logits / temperature)
kl = F.kl_div(student_log_probs, teacher_probs)
return temperature ** 2 * kl
```

解释：

- teacher logits 先除以 temperature，分布更软。
- student 学的不只是正确 token，也学 teacher 对其他 token 的概率判断。
- 乘 `temperature ** 2` 是蒸馏常见缩放，保持梯度量级。

#### 5.8.2 蒸馏训练 loss

```python
ce_loss = cross_entropy(student_logits, labels, ignore_index=-100)
distill_loss = distillation_loss(student_logits[mask], teacher_logits[mask])
loss = alpha * ce_loss + (1 - alpha) * distill_loss
```

这里 mask 来自 SFT labels，主要优化 assistant 区间。

### 5.9 Agent RL：工具调用闭环

Agent 代码在 `trainer/train_agent.py`。

#### 5.9.1 `parse_tool_calls()`

```python
for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
    calls.append(json.loads(m.strip()))
```

它从模型输出中解析工具调用 JSON。

#### 5.9.2 `execute_tool()`

```python
fn = MOCK_RESULTS.get(name)
return fn(args)
```

它根据工具名执行模拟工具，比如计算、天气、时间、汇率、翻译。

代码设置了 alarm timeout，避免工具执行卡住。

#### 5.9.3 `rollout_single()`

这是 Agent multi-turn 的核心。

每一轮：

1. 用 chat template 渲染 messages 和 tools。
2. rollout 生成 assistant 输出。
3. 解析 `<tool_call>`。
4. 如果没有工具调用，结束。
5. 如果有工具调用，执行工具。
6. 把 `<tool_response>` 追加到 messages。
7. 继续下一轮生成。

同时它会记录：

```text
prompt_ids
response_ids
response_mask
response_old_logps
turn_outputs
unfinished
```

其中 `response_mask=0` 的工具观察 token 不作为 policy 直接优化目标。

#### 5.9.4 Agent `calculate_rewards()`

如果没有工具调用：

- 长度奖励。
- thinking 奖励。
- reward model 分数。
- 重复惩罚。

如果有工具调用：

- tool_call 标签不匹配扣分。
- 工具名必须在合法工具列表。
- 参数必须通过 `CHECK_ARGS`。
- 工具调用数量要接近 gt 数量。
- 最终答案命中 gt 加分。
- 多轮未完成扣分。

这说明 Agent RL 优化的不只是回答质量，还包括“会不会正确调用工具”。

### 5.10 每个训练方法的代码阅读顺序

建议你以后按这个顺序读代码：

| 方法 | 阅读顺序 |
| --- | --- |
| Pretrain | `PretrainDataset` -> `train_pretrain.py::train_epoch` -> `MiniMindForCausalLM.forward` |
| Full SFT | `SFTDataset.generate_labels` -> `train_full_sft.py::train_epoch` |
| LoRA | `LoRA` -> `apply_lora` -> 冻结参数逻辑 -> `save_lora/merge_lora` |
| DPO | `DPODataset` -> `logits_to_log_probs` -> `dpo_loss` -> `train_epoch` |
| PPO | `RLAIFDataset` -> `rollout_engine` -> `calculate_rewards` -> `ppo_train_epoch` |
| GRPO | `RLAIFDataset` -> `rollout_engine` -> `calculate_rewards` -> `grpo_train_epoch` |
| Distillation | `SFTDataset` -> `distillation_loss` -> student/teacher forward |
| Agent RL | tools schema -> `rollout_single` -> `execute_tool` -> Agent `calculate_rewards` |

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
