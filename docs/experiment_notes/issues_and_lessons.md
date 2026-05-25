# MiniMind/TinyChatLM 实验问题复盘与面试追问

## 核心结论

这次实验最重要的价值不是“模型已经很强”，而是跑通了从 Pretrain、Full SFT、LoRA SFT 到 DPO、PPO、GRPO 的完整训练链路，并且能用指标解释每个阶段的变化。

**最关键的量化证据**是：在相同 SFT-style 验证集上，Full SFT 的 PPL 从 Pretrain 的 105.8949 降到 13.6782。这说明 SFT 不只是“继续训练”，而是显著提升了模型对 user-assistant 对话格式和 assistant 回答分布的建模能力。

同时，PPO / GRPO 阶段也暴露出小模型强化对齐的典型问题：reward 偏低、生成容易到最大长度、策略更新较弱。这些现象不代表链路失败，反而说明后续需要继续调 reward model、KL、学习率、生成长度和数据质量。

## 问题与解决方案

### 1. 模型参数量和预期不一致

实验中先后观察到：

| 配置 | 参数量 | 说明 |
| --- | ---: | --- |
| `hidden_size=512, layers=8` | 25.83M | 小规模 smoke test |
| `hidden_size=768, layers=8` | 54.47M | 正式复现配置 |

一开始预期是 64M，但实际参数量由 `hidden_size`、`num_hidden_layers`、`vocab_size`、权重共享、是否 MoE 等共同决定。当前 `hidden_size=768, num_hidden_layers=8` 是 50M-60M 级配置，不会严格等于 64M。

解决方案：训练、SFT、DPO、PPO、GRPO、eval 阶段都显式写：

```bash
--hidden_size 768
--num_hidden_layers 8
```

实验记录中使用更严谨表述：

> 复现 50M-60M 级 MiniMind 小模型训练链路，使用 `hidden_size=768`、`num_hidden_layers=8` 配置，实测约 54.47M 参数。

### 2. SwanLab 参数名容易混淆

代码参数叫：

```bash
--use_wandb
--wandb_project
```

但代码实际导入的是：

```python
import swanlab as wandb
```

因此这里的 `wandb` 变量只是兼容命名，本质是 SwanLab。预训练、SFT、DPO、PPO、GRPO 都通过 `wandb.log(...)` 记录 loss、学习率、reward、KL 等指标。

解决方案：启动命令仍使用：

```bash
--use_wandb
--wandb_project TinyChatLM-xxx
```

但实验记录中说明：

```text
use_wandb = 开启 SwanLab
wandb_project = SwanLab 项目名
```

### 3. LoRA 默认训练配置过重

LoRA 脚本默认训练轮数和保存频率偏重，第一次验证链路容易耗时过长。LoRA 默认从 `full_sft` 加载，如果想从 `pretrain` 直接 LoRA，需要显式传 `--from_weight pretrain`，但效果通常不如标准路线。

解决方案：第一次验证链路建议：

```bash
--epochs 1
--batch_size 8
--save_interval 100
```

正式实验可以先试：

```bash
--epochs 3
```

推荐训练路线：

```text
pretrain -> full_sft -> lora_sft
```

### 4. 从 pretrain 直接 LoRA 效果不好

从 `pretrain` 直接接 LoRA SFT 后，模型虽然能生成，但容易出现回答开头标点异常、代码生成不稳定、重复和答非所问。

原因是：

```text
pretrain 模型主要学续写
full_sft 模型才学用户问题 -> 助手回答
```

解决方案：如果只是验证 LoRA 代码链路，可以跑 `pretrain -> LoRA`；如果追求回答效果，应使用 `pretrain -> full_sft -> LoRA`。

### 5. 路径问题频繁出现

常见错误包括：

```text
FileNotFoundError
ModuleNotFoundError: No module named 'model'
找不到 ../dataset/xxx.jsonl
找不到 ./out/full_sft_768.pth
```

本质是运行目录不一致：

```text
在项目根目录运行：路径写 ./dataset/xxx.jsonl、./out/xxx.pth
在 trainer 目录运行：路径写 ../dataset/xxx.jsonl、../out/xxx.pth
```

解决方案：固定运行规则：

```text
训练脚本：在 trainer 目录里跑
eval_llm.py：在项目根目录跑
数据路径：trainer 里用 ../dataset/xxx.jsonl
权重路径：trainer 里默认 ../out，根目录里默认 ./out
```

每次实验必须记录完整命令，而不是只写“跑了 SFT”。

### 6. Hugging Face 下载和依赖版本冲突

升级依赖后出现过：

```text
transformers requires huggingface-hub<1.0
swanlab requires rich<14.0.0
Network is unreachable
```

解决方案：先固定兼容版本：

```bash
pip install "huggingface_hub==0.36.2" "rich==13.7.1"
```

下载模型或数据时优先使用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download ...
```

或直接使用：

```bash
wget https://hf-mirror.com/...
```

### 7. `OMP_NUM_THREADS` 环境变量非法

报错：

```text
libgomp: Invalid value for environment variable OMP_NUM_THREADS
```

这不是模型错误，而是环境变量值非法。

解决方案：

```bash
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

### 8. PPO / GRPO 需要 Reward Model

PPO 和 GRPO 不是只用 SFT 数据。它们需要 prompt 数据和外部 reward model。

PPO 训练中涉及：

- actor model
- critic model
- reference model
- reward model
- rollout engine

需要记录的指标包括 `reward`、`KL_ref`、`approx_kl`、`clipfrac`、`critic_loss`、`avg_response_len`、`actor_lr`、`critic_lr`。

GRPO 也需要：

```bash
--reward_model_path
--data_path
```

如果使用普通模型，需要确保推理/训练脚本不要加载 reasoning 权重。若脚本存在 `reasoning` 参数，应显式设置为普通模式，避免误找 `reason_768.pth`。

### 9. SwanLab 云端 500

GRPO 训练时出现：

```text
api.swanlab.cn too many 500 error responses
```

这不是 GRPO 代码错误，而是 SwanLab 云端接口异常。

解决方案：先去掉：

```bash
--use_wandb
```

本地跑通训练；等 SwanLab 服务恢复后再开启日志上报。

## 量化指标记录

### 模型规模

| 配置 | 参数量 | 用途 |
| --- | ---: | --- |
| `hidden_size=512, layers=8` | 25.83M | 快速 smoke test |
| `hidden_size=768, layers=8` | 54.47M | 正式复现配置 |

### Pretrain vs Full SFT PPL

在相同 SFT-style 验证集上评估：

| 模型 | avg_loss | PPL | valid_tokens |
| --- | ---: | ---: | ---: |
| Pretrain | 4.6624 | 105.8949 | 685 |
| Full SFT | 2.6158 | 13.6782 | 685 |

当前验证集规模为 10 examples / 685 valid tokens，因此适合作为流程验证和初步对比。更正式的实验建议扩展到 100-500 条固定 eval 样本，并确保 eval 集不参与训练。

### 训练 loss 指标

Pretrain / SFT / LoRA 阶段重点记录：

| 指标 | 含义 |
| --- | --- |
| `loss` | 总损失 |
| `logits_loss` | next-token prediction 交叉熵 |
| `aux_loss` | MoE 辅助损失，不开 MoE 时为 0 |
| `learning_rate` | 当前学习率 |
| `epoch_time` | 当前 epoch 剩余时间估计 |

注意：loss 下降只能说明 token-level 拟合更好，不等价于生成质量一定更好。

### DPO 指标

DPO 当前已记录：

```text
loss
dpo_loss
aux_loss
learning_rate
epoch_time
```

DPO 核心计算是：

```text
logits = pi_logratios - ref_logratios
loss = -logsigmoid(beta * logits)
```

后续建议扩展：

| 指标 | 含义 |
| --- | --- |
| `preference_accuracy` | policy 是否更偏好 chosen |
| `win_rate` | chosen logp 是否大于 rejected logp |
| `reward_margin` | chosen 和 rejected 的偏好差距 |
| `dpo_loss` | DPO 优化目标 |

### PPO 指标

PPO 阶段建议记录：

| 指标 | 含义 |
| --- | --- |
| `actor_loss` | 策略模型更新损失 |
| `critic_loss` | value 模型估计误差 |
| `reward` | Reward Model 给生成回答的平均分 |
| `KL / KL_ref` | 当前策略偏离 old policy / reference 的程度 |
| `avg_response_len` | 平均生成长度 |
| `actor_lr / critic_lr` | actor 和 critic 学习率 |

当前观察：reward 多数为负，`avg_response_len` 经常达到最大长度。这说明 PPO 链路已经跑通，但回答质量、停止能力和 reward 设计还不稳定。

### GRPO 指标

GRPO 阶段观察到：

```text
Actor Loss: 0.0000
Reward: -1.8667 / -2.0439 / -3.0000
Avg Response Len: 128.00
Learning Rate: 7e-8
```

| 指标 | 当前现象 | 解释 |
| --- | --- | --- |
| Actor Loss | 接近 0 | 更新较弱，可能和 learning rate 小、reward 方差不足、num_generations 或 loss 缩放有关 |
| Reward | 多数为负 | Reward Model 认为生成质量不高 |
| Avg Response Len | 有时达到 128 | 模型可能生成到最大长度才停 |
| Learning Rate | 约 `7e-8` | RL 阶段学习率很小，用于防止训崩 |

### 生成样例对比

建议固定以下 prompt 做 Pretrain / SFT / DPO / PPO / GRPO 对比：

| Prompt | Pretrain 观察点 | SFT 观察点 | DPO / PPO / GRPO 观察点 |
| --- | --- | --- | --- |
| 你有什么特长？ | 是否有续写感 | 是否能按助手身份回答 | 是否更安全、更简洁 |
| 为什么天空是蓝色的？ | 是否重复或事实不稳 | 是否提到散射 | 是否减少重复 |
| 请用 Python 写斐波那契函数 | 是否能稳定生成代码 | 是否仍较弱 | 是否有结构改善 |
| 解释光合作用 | 是否泛化空洞 | 是否能给出基本解释 | 是否更完整 |
| 推荐中国美食 | 是否列表混乱 | 是否更像助手推荐 | 是否更有条理 |

## 实验认知总结

### 1. Pretrain 和 SFT 的区别可以量化

PPL 从 105.8949 降到 13.6782，说明 SFT 确实让模型适应了问答格式。

可以总结为：

> Pretrain 学语言分布和续写能力，SFT 学对话格式和指令跟随能力。

### 2. loss 下降不等于回答一定好

LoRA loss 下降后，生成仍可能出现开头标点、重复、答非所问。这说明训练 loss 只是 token-level 拟合指标，生成质量还要看样例、格式正确率、事实准确性、停止能力和重复率。

### 3. 小模型更适合证明训练链路

54M 级小模型可以学到基本问答格式，但在代码生成、复杂事实解释、长回答一致性和停止生成上仍不稳定。

这类复现实验的重点应是：

- 数据格式是否正确。
- loss mask 是否正确。
- 训练阶段是否衔接。
- 偏好对齐和 RL 对齐流程是否跑通。
- 指标是否能解释现象。

### 4. DPO / PPO / GRPO 的差异更清楚

| 方法 | 依赖数据 | 是否需要 Reward Model | 主要指标 |
| --- | --- | --- | --- |
| DPO | chosen / rejected 偏好数据 | 不需要外部 RM | `dpo_loss`、`preference_accuracy` |
| PPO | prompt 数据 + Reward Model | 需要 | `reward`、`KL`、`actor_loss`、`critic_loss` |
| GRPO | prompt 数据 + 多个生成 + Reward Model | 需要 | `reward`、`actor_loss`、`avg_response_len` |

DPO 更简单稳定；PPO / GRPO 更复杂，更吃显存，也更容易出现 reward 低、长度失控、更新弱的问题。

### 5. 路径、参数和权重一致性非常重要

每次实验都必须记录：

```text
运行目录
完整命令
权重名
hidden_size
num_hidden_layers
data_path
use_moe
reasoning
reward_model_path
```

否则很容易出现加载错权重、参数量不一致、路径找不到、评估结果不可复现等问题。

## 面试可能追问

### Q1: 为什么 `hidden_size=768` 不是严格 64M？

回答要点：参数量由 embedding/lm_head、attention、FFN、norm、层数、词表大小、权重共享共同决定。当前权重共享后，`hidden_size=768, layers=8` 实测约 54.47M，是 50M-60M 级小模型配置，不应硬写严格 64M。

### Q2: Pretrain 在 SFT 验证集上 PPL 很高，是否说明模型没学会中文？

回答要点：不是。Pretrain 学的是通用 next-token continuation，SFT 验证集是 user-assistant 对话格式。PPL 高说明它不适应 assistant 回答分布，不代表完全不会中文。

### Q3: 为什么 SFT 后 PPL 能显著下降？

回答要点：SFT 使用 chat template 和 assistant-only loss，模型看到 user/system 上下文，但只优化 assistant 回复 token。因此它更会预测“用户提问后助手应该怎么回答”，PPL 会在 SFT-style 验证集上下降。

### Q4: assistant-only loss 有什么作用？

回答要点：它避免模型学习生成用户问题或 system prompt，只让模型学习 assistant 回复。这样训练目标和推理目标一致：推理时给定用户输入，模型只需要生成 assistant 内容。

### Q5: 为什么 loss 降了但生成仍可能重复或答非所问？

回答要点：loss 是 token-level 拟合指标，不能完全代表开放生成质量。重复、停止能力、事实准确性、格式稳定性还受数据质量、解码参数、模型容量和对齐训练影响。

### Q6: 为什么 LoRA 更推荐接在 `full_sft` 后，而不是直接接 `pretrain`？

回答要点：pretrain 主要具备续写能力，尚未学好对话格式。LoRA 参数量少，适合在已有能力上做增量适配；如果从 pretrain 直接 LoRA，adapter 既要学对话格式又要学任务能力，难度更高，效果更不稳定。

### Q7: DPO 和 PPO/GRPO 的本质区别是什么？

回答要点：DPO 是离线偏好优化，直接用 chosen/rejected 和 frozen reference model 优化 logprob margin，不需要 reward model 和在线 rollout。PPO/GRPO 是在线生成后用 reward model 打分，再用 RL 目标更新 policy。

### Q8: 为什么 DPO 不需要 Reward Model？

回答要点：DPO 把偏好数据本身转化为优化目标，通过比较 policy 和 reference 在 chosen/rejected 上的 log ratio 来学习偏好，因此不需要显式训练或加载 reward model。

### Q9: PPO 里 reward 为负、平均长度达到最大值说明什么？

回答要点：说明 reward model 对生成质量评分较低，并且模型停止能力不好，可能生成到 `max_gen_len` 才停。后续应调小 max generation length、优化 eos 训练、改 reward、控制 KL 和学习率。

### Q10: GRPO 的 actor loss 接近 0 可能意味着什么？

回答要点：可能是学习率太小、组内 reward 方差不足、advantage 标准化后信号弱、ratio/clipping 让更新很小，或生成质量普遍差导致 reward 区分度不足。需要结合 reward、KL、response length 和样例一起看。

### Q11: 为什么 RL 阶段要监控 KL、reward、response length，而不是只看 loss？

回答要点：RL loss 的绝对值不一定直接对应模型质量。reward 表示优化目标，KL 表示策略偏离程度，response length 能暴露长度投机和停止失败。只看 loss 容易漏掉 reward hacking 或策略崩掉。

### Q12: 为什么运行目录、权重名、`hidden_size`、`use_moe`、`reasoning` 必须一致？

回答要点：这些参数决定模型结构和权重路径。结构参数不一致会加载失败或 silently mismatch；路径不一致会找不到数据或权重；`use_moe/reasoning` 错误会让脚本找错 checkpoint，导致实验不可复现。

## 后续改进方向

- 扩大 SFT eval 集到 100-500 条，并固定不参与训练。
- 为 DPO 增加 `preference_accuracy`、`win_rate`、`reward_margin`。
- PPO/GRPO 降低 `max_gen_len`，观察平均长度和 reward 是否改善。
- 调整 KL 系数、学习率和 reward model，避免 reward 偏低和策略更新过弱。
- 固定一组生成 prompt，持续对比 Pretrain、SFT、LoRA、DPO、PPO、GRPO 的输出。
- 为每次实验保存完整命令、配置、权重来源、数据路径和日志截图。

## 简历 / 面试总结版本

可以这样表述：

> 我复现了 MiniMind/TinyChatLM 的轻量中文小模型训练链路，包括 Pretrain、Full SFT、LoRA SFT、DPO、PPO 和 GRPO。实验中我使用 SwanLab 记录 loss、learning rate、reward、KL、response length 等指标，并构造固定 SFT 验证集计算 PPL。结果显示，Full SFT 在相同验证集上的 PPL 从 Pretrain 的 105.89 降至 13.68，说明 SFT 显著增强了模型对 user-assistant 对话格式的建模能力。在 PPO / GRPO 阶段，我进一步分析了 reward、KL 和平均生成长度，发现小模型在强化学习对齐中容易出现 reward 偏低、生成过长和策略更新较弱的问题，因此需要控制 learning rate、KL 系数、max generation length 和 reward model 质量。

最核心的一句话：

> 这次实验的价值不是模型已经很强，而是跑通了从预训练到指令微调、偏好对齐、强化学习对齐的完整链路，并能用 PPL、loss、reward、KL、生成长度和样例对比解释每个阶段的效果。
