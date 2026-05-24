# LoRA SFT 学习笔记

## LoRA 的目标

LoRA 用少量低秩参数适配模型，而不是更新全部权重。它适合小显存、快速实验和多任务 adapter 管理。

当前 `train_lora.py` 默认从 `full_sft` 权重开始训练，说明它被设计成“在已有对话模型上做领域或身份适配”的阶段。

## 低秩结构

`model/model_lora.py` 中的 LoRA 模块包含两个线性层：

```text
A: in_features -> rank
B: rank -> out_features
LoRA(x) = B(A(x))
```

注入后，原线性层输出变成：

```text
Linear(x) + LoRA(x)
= W x + B(Ax)
```

这相当于给原权重增加一个低秩增量：

```text
W' = W + B @ A
```

rank 越大，可训练容量越强，但参数量和显存占用也越高。

## 初始化设计

代码中 A 使用高斯初始化，B 使用全零初始化：

```text
A.weight ~ Normal(0, 0.02)
B.weight = 0
```

这样训练刚开始时 `B(Ax)=0`，模型输出和原模型完全一致。这个设计很重要：LoRA adapter 初始不会破坏已经 SFT 好的基座模型，训练只是在原行为基础上逐步学习增量。

## 注入策略

`apply_lora(model, rank=16)` 会遍历所有 `nn.Linear`，只给输入输出维度相等的方阵线性层添加 LoRA：

```text
if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]
```

这种实现简单，适合教学复现。但它和常见 PEFT 配置中的 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` 白名单方式不同。当前代码只注入方阵，可能覆盖 attention 的部分 projection，但不一定覆盖 FFN 中所有非方阵投影。

文档和实验报告中应如实表述：这是一个原生 monkey-patch LoRA 实现，便于理解低秩增量机制；如果后续要做严格 LoRA target modules 对比，需要把注入策略改成显式模块名匹配。

## 参数冻结

`train_lora.py` 在注入 LoRA 后遍历全部参数：

- 名字包含 `lora` 的参数：`requires_grad=True`
- 其他参数：`requires_grad=False`

优化器只接收 `lora_params`，梯度裁剪也只裁剪 LoRA 参数。这体现了参数高效微调的核心：基座模型作为固定能力底座，adapter 学习任务增量。

## 保存、加载与合并

`save_lora` 只保存 `.lora.` 子模块的 state dict，并转成 half 后落盘。这让 adapter 文件远小于完整模型。

`load_lora` 从文件里取出各模块对应的 LoRA 权重，加载到已经注入 LoRA 的模型上。

`merge_lora` 先加载 adapter，再把 `B @ A` 加到原线性层权重上，保存不含 `.lora.` 的完整权重。合并后的模型推理时不再需要额外 LoRA 分支，部署更简单。

## train_lora.py 默认配置

| 参数 | 默认值 | 意义 |
| --- | --- | --- |
| `lora_name` | `lora_medical` | adapter 保存名 |
| `from_weight` | `full_sft` | 从全参 SFT 模型开始 |
| `epochs` | 10 | LoRA 训练轮数 |
| `batch_size` | 32 | batch size |
| `learning_rate` | `1e-4` | adapter 学习率 |
| `max_seq_len` | 340 | 训练长度 |
| `data_path` | `../dataset/lora_medical.jsonl` | LoRA 数据 |

脚本会打印总参数量、LoRA 参数量和参数占比，这是 LoRA 实验最应该保留的证据。

## Rank 对比模板

| Rank | LoRA 参数量 | 参数占比 | 峰值显存 | tokens/sec | Train loss | Eval PPL/评分 | 样例结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 16 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 32 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

简历表达上，LoRA 这部分可以强调：你不只是会调用 PEFT，而是理解 adapter 本质是对原权重的低秩增量，并实现了注入、冻结、保存和合并流程。
