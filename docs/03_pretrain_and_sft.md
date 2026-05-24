# Pretrain 与 Full SFT 学习笔记

## 两个阶段的目标差异

Pretrain 学的是语言建模能力：给定前文预测下一个 token。它的典型表现是续写，例如补全百科、故事、代码或普通文本。

Full SFT 学的是对话和指令跟随：给定 system/user 上下文，生成 assistant 回答。它的典型表现是遵循问题、保持角色格式、输出更像助手。

因此，Pretrain 和 SFT 都可以用 causal LM loss，但训练数据和 label mask 不同。Pretrain 对完整文本的非 pad token 做 next-token prediction；SFT 只对 assistant 区间计算 loss，避免模型学习“如何生成用户问题”。

## train_pretrain.py

`train_pretrain.py` 的默认关键参数：

| 参数 | 默认值 | 意义 |
| --- | --- | --- |
| `save_weight` | `pretrain` | 输出权重前缀 |
| `epochs` | 2 | 训练轮数 |
| `batch_size` | 32 | 每步 batch size |
| `learning_rate` | `5e-4` | 初始学习率 |
| `dtype` | `bfloat16` | 混合精度类型 |
| `accumulation_steps` | 8 | 梯度累积步数 |
| `max_seq_len` | 340 | 训练截断长度 |
| `from_weight` | `none` | 默认从头训练 |
| `data_path` | `../dataset/pretrain_t2t_mini.jsonl` | 预训练数据 |

训练过程调用 `PretrainDataset` 返回 `(input_ids, labels)`。Dataset 会读取样本中的 `text` 字段，构造 `bos + text_tokens + eos + pad`，并把 pad 位置的 label 设为 `-100`。训练脚本再执行：

```text
res = model(input_ids, labels=labels)
loss = res.loss + res.aux_loss
```

其中 `res.loss` 是 causal LM cross entropy，`res.aux_loss` 只有 MoE 模型会产生非零值。

## train_full_sft.py

`train_full_sft.py` 的默认关键参数：

| 参数 | 默认值 | 意义 |
| --- | --- | --- |
| `save_weight` | `full_sft` | 输出权重前缀 |
| `learning_rate` | `1e-5` | SFT 学习率，比 pretrain 小 |
| `batch_size` | 16 | 默认 batch size |
| `accumulation_steps` | 1 | 默认不累积 |
| `max_seq_len` | 768 | 对话训练长度 |
| `from_weight` | `pretrain` | 从预训练权重开始 |
| `data_path` | `../dataset/sft_t2t_mini.jsonl` | SFT 数据 |

SFT 同样调用 `model(input_ids, labels=labels)`。区别在于 `SFTDataset.generate_labels` 会先把所有 label 置为 `-100`，再扫描 `<|im_start|>assistant\n` 到 `<|im_end|>\n` 的 token 区间，只对 assistant 回复和结束标记计算 loss。

`SFTDataset` 还会做两类数据增强：如果普通对话没有 system message，会以 20% 概率插入中英文 system prompt；如果渲染后出现空 thinking 标签，会以 80% 概率移除，避免模型强制学习每次都输出空 think。

## 统一训练工程组件

两个脚本共用了一套训练工程模式。

DDP 通过 `init_distributed_mode()` 判断环境变量 `RANK` 是否存在。如果是分布式训练，使用 NCCL 初始化，并把模型包成 `DistributedDataParallel`。RoPE buffer `freqs_cos` 和 `freqs_sin` 被加入 `_ddp_params_and_buffers_to_ignore`，避免不必要同步。

混合精度通过 autocast 控制。`dtype=bfloat16` 时使用 bf16 autocast，`dtype=float16` 时启用 GradScaler。bf16 的优势是指数范围更大，通常比 fp16 更稳；fp16 则需要 scaler 避免梯度下溢。

梯度累积通过 `loss / accumulation_steps` 和每隔 N step 执行 optimizer step 实现。它的意义是用较小显存模拟更大的 global batch。

梯度裁剪通过 `clip_grad_norm_(model.parameters(), grad_clip)` 实现，默认阈值 1.0，用于限制异常梯度导致的训练不稳定。

学习率由 `get_lr` 给出：

```text
lr * (0.1 + 0.45 * (1 + cos(pi * current_step / total_steps)))
```

这相当于从初始 lr 平滑衰减到约 `0.1 * lr`。函数名虽然不包含 warmup，但代码实际是 cosine decay，没有显式 warmup 阶段。

checkpoint 分两类：

- `../out/<weight>_<hidden_size>.pth`：半精度模型权重。
- `../checkpoints/<weight>_<hidden_size>_resume.pth`：包含模型、优化器、epoch、step、world size、wandb id 等恢复状态。

`SkipBatchSampler` 支持从中断 step 继续训练。如果恢复时 GPU 数变化，`lm_checkpoint` 会按 world size 调整 step。

## Pretrain vs SFT 对比观察点

后续 out 对比可以重点看：

| 对比点 | Pretrain | Full SFT |
| --- | --- | --- |
| 训练目标 | 文本续写 | assistant 响应 |
| 默认起点 | 随机初始化 | `pretrain` |
| 默认学习率 | `5e-4` | `1e-5` |
| 默认长度 | 340 | 768 |
| 典型输出 | 继续写文本 | 回答用户问题 |
| 关键风险 | loss 降但不会聊天 | 过拟合格式或遗忘基础能力 |

简历表达上，这组对比可以说明你理解了“语言模型预训练”和“指令微调”不是同一个任务：二者共享 causal LM 形式，但数据组织、loss mask 和行为目标不同。
