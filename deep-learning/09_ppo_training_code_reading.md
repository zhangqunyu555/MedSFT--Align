# MedicalGPT 训练源码精读 09：ppo_training.py

## 整体作用

`MedicalGPT/training/ppo_training.py` 文件名叫 PPO，但当前源码开头已经写明：

```text
Train a model from SFT using RLOO (REINFORCE Leave-One-Out, PPO alternative)
```

也就是说，这个脚本并不是经典 PPOTrainer 实现，而是用 TRL 的 `RLOOTrainer` 做 PPO 替代式 RLHF 训练。

它的作用是：

```text
SFT 后的 policy model
  -> 根据 prompt 生成回答
  -> reward model 给回答打分
  -> RLOOTrainer 根据奖励更新 policy
```

## 方法原理

经典 PPO/RLHF 通常有三类模型：

- `policy model`：当前要训练的模型。
- `reference model`：冻结参考模型，用 KL 约束防止训飞。
- `reward model`：给回答打分。

PPO 的核心直觉是：如果某个回答 reward 高，就提高它的生成概率；如果 reward 低，就降低它的生成概率。同时用 KL 控制 policy 不要偏离原模型太远。

RLOO 是 REINFORCE Leave-One-Out。它不使用单独的 value model，而是对同一个 prompt 采样多个回答，用组内其他回答的平均 reward 作为 baseline，降低方差。MedicalGPT 这里用的是 `RLOOTrainer`：

```python
from trl import RLOOConfig, RLOOTrainer
```

所以读这个文件时要记住：

```text
文件名 ppo_training.py
实际训练器 RLOOTrainer
定位 PPO 替代式 RLHF
```

## 脚本输入输出

输入：

```text
--sft_model_path outputs/qwen3_4b_medical_qlora_381k_then_top100k
--reward_model_path outputs/qwen3_4b_medical_reward_model
--train_file_dir data/rl_prompts
--template_name qwen3
```

训练数据需要至少有 `conversations`，脚本会从里面抽取 user prompt。

输出：

```text
outputs/qwen3_4b_medical_rloo/
  adapter 或模型权重
  trainer_state.json
```

## 主流程图

```text
解析 RLOOArguments / RLOOConfig / ModelConfig
  -> 判断单卡、多卡、DDP
  -> 加载 tokenizer
  -> 加载 reward model
  -> 加载 policy model
  -> 加载 prompt 数据
  -> ShareGPT conversations 转 prompt
  -> RLOOTrainer 训练
  -> 保存模型
  -> generate_completions 查看样例输出
```

## 关键源码精读

### 参数类：RLOOArguments

#### 源码

```python
class RLOOArguments:
    """
    The name of the Casual LM model we wish to fine with RLOO
    """
    sft_model_path: Optional[str] = field(default=None, metadata={"help": "Path to the SFT model."})
    reward_model_path: Optional[str] = field(default=None, metadata={"help": "Path to the reward model."})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field(default=None, metadata={"help": "Dataset configuration name."})
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )
    template_name: Optional[str] = field(
        default=None,
        metadata={"help": "The prompt template name. If not set, use tokenizer's built-in chat_template."}
    )
    max_source_length: Optional[int] = field(default=1024, metadata={"help": "Max length of prompt input text"})
```

#### 逐段解释

`sft_model_path` 是策略模型起点，应该指向 SFT 后的模型或 adapter 合并后的模型。

`reward_model_path` 是奖励模型路径。这个脚本需要一个 sequence classification 模型：

```python
AutoModelForSequenceClassification.from_pretrained(...)
```

也就是说，不能直接拿普通 CausalLM 当 reward model，除非它已经按 reward modeling 脚本训练成分类/打分模型。

`train_file_dir` 和 `validation_file_dir` 是本地 prompt 数据。

`template_name=qwen3` 保证 prompt 和前面 SFT 模板一致。

### 解析参数和设备判断

#### 源码

```python
def main():
    parser = HfArgumentParser((RLOOArguments, RLOOConfig, ModelConfig))
    args, training_args, model_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )[:3]

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = local_rank == 0
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    num_gpus = torch.cuda.device_count()

    torch_dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )

    if ddp:
        device_map = None
        max_memory = None
    elif num_gpus > 1:
        max_memory = {}
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            total_mem = gpu_props.total_memory
            usable_mem = int(total_mem * 0.8)
            max_memory[i] = f"{usable_mem // (1024 ** 2)}MiB"
        device_map = "auto"
    else:
        device_map = "auto"
        max_memory = None
```

#### 逐段解释

这个脚本同时解析三类参数：

- `RLOOArguments`：数据、SFT 模型、reward 模型。
- `RLOOConfig`：TRL 训练参数。
- `ModelConfig`：模型加载和 PEFT 相关参数。

设备判断分三种：

- DDP 多进程：`device_map=None`，让分布式框架接管。
- 单进程多卡：`device_map=auto`，并为每张卡预留 20% 显存。
- 单卡：`device_map=auto`。

#### 为什么要预留显存

RL 训练比 SFT 更吃显存，因为它需要生成、计算 reward、保存 logprobs 等中间量。预留 20% 能减少 OOM。

### 加载 tokenizer

#### 源码

```python
sft_model_path = args.sft_model_path or model_args.model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(
    sft_model_path, trust_remote_code=model_args.trust_remote_code
)
if tokenizer.eos_token_id is None:
    tokenizer.eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.sep_token
    tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
if tokenizer.bos_token_id is None:
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id
if tokenizer.pad_token_id is None:
    if tokenizer.unk_token_id is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.eos_token
```

#### 逐段解释

tokenizer 优先从 `sft_model_path` 加载。因为 RL 阶段要延续 SFT 的 tokenizer 和 chat 模板，不应该随便换。

补齐 `eos/bos/pad` 的逻辑和 SFT 类似。

### 加载 reward model

#### 源码

```python
reward_model_kwargs = dict(
    trust_remote_code=model_args.trust_remote_code,
    num_labels=1,
)
if torch_dtype is not None:
    reward_model_kwargs["torch_dtype"] = torch_dtype
if device_map is not None:
    reward_model_kwargs["device_map"] = device_map
if max_memory is not None:
    reward_model_kwargs["max_memory"] = max_memory

reward_model = AutoModelForSequenceClassification.from_pretrained(
    args.reward_model_path, **reward_model_kwargs
)
```

#### 逐段解释

奖励模型是 `AutoModelForSequenceClassification`，并且 `num_labels=1`，说明它输出一个标量分数。

RLHF 中 reward model 的输入通常是：

```text
prompt + response
```

输出是一个分数：

```text
回答质量越高 -> reward 越高
```

#### 和当前项目的关系

如果你要用这个脚本做“复杂病例格式回答准确率 72% -> 94%”，需要先有一个 reward model。它可以用偏好数据训练，例如：

```text
格式正确、医学准确 -> 高分
格式错误、医学错误 -> 低分
```

否则这个 PPO/RLOO 脚本不能直接跑。

### 加载 policy model

#### 源码

```python
policy_kwargs = dict(
    trust_remote_code=model_args.trust_remote_code,
)
if torch_dtype is not None:
    policy_kwargs["torch_dtype"] = torch_dtype
if device_map is not None:
    policy_kwargs["device_map"] = device_map
if max_memory is not None:
    policy_kwargs["max_memory"] = max_memory

policy = AutoModelForCausalLM.from_pretrained(
    sft_model_path, **policy_kwargs
)

peft_config = get_peft_config(model_args)
```

#### 逐段解释

policy 是真正要优化的语言模型。它从 SFT 模型初始化，而不是从原始 Qwen3 初始化。

`get_peft_config(model_args)` 是 TRL 提供的 PEFT 配置工具，会根据命令行参数决定是否使用 LoRA。

### 数据加载

#### 源码

```python
prompt_template = None
if args.template_name:
    prompt_template = get_conv_template(args.template_name)
if args.dataset_name is not None:
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_train_split
    )
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
else:
    data_files = {}
    if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
        train_data_files = glob(f'{args.train_file_dir}/**/*.jsonl', recursive=True)
        data_files["train"] = train_data_files
    if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
        eval_data_files = glob(f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
        data_files["validation"] = eval_data_files
    dataset = load_dataset(
        'json',
        data_files=data_files,
    )
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    eval_dataset = val_dataset.select(range(min(100, len(val_dataset))))
```

#### 逐段解释

本地数据使用 HuggingFace datasets 的 json loader。

RL 评估只取 validation 前 100 条：

```python
eval_dataset = val_dataset.select(range(min(100, len(val_dataset))))
```

这是为了节省 RL 训练时的评估成本。

### prompt 预处理

#### 源码

```python
def preprocess_function(examples):
    new_examples = {"prompt": []}
    roles = ["human", "gpt"]

    def get_dialog(examples):
        system_prompts = examples.get("system_prompt", "")
        for i, source in enumerate(examples['conversations']):
            if len(source) < 2:
                continue
            data_role = source[0].get("from", "")
            if data_role not in roles or data_role != roles[0]:
                source = source[1:]
            if len(source) < 2:
                continue
            messages = []
            for j, sentence in enumerate(source):
                data_role = sentence.get("from", "")
                if data_role not in roles:
                    logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                    break
                if data_role == roles[j % 2]:
                    messages.append(sentence["value"])
            if len(messages) < 2 or len(messages) % 2 != 0:
                continue
            history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
            system_prompt = system_prompts[i] if system_prompts else None
            if prompt_template:
                yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)
            else:
                convs = []
                accumulated = []
                if system_prompt:
                    accumulated.append({"role": "system", "content": system_prompt})
                prev_text = ""
                for uq, br in history_messages:
                    accumulated.append({"role": "user", "content": uq})
                    cur_text = tokenizer.apply_chat_template(
                        accumulated, tokenize=False, add_generation_prompt=True
                    )
                    convs.append(cur_text[len(prev_text):])
                    convs.append(br)
                    accumulated.append({"role": "assistant", "content": br})
                    prev_text = tokenizer.apply_chat_template(
                        accumulated, tokenize=False, add_generation_prompt=False
                    )
                yield convs

    for dialog in get_dialog(examples):
        for i in range(len(dialog) // 2):
            source_txt = dialog[2 * i]
            new_examples["prompt"].append(source_txt)

    return new_examples
```

#### 逐段解释

这个预处理函数和 SFT 的最大区别是：它只保留 prompt，不保留答案。

RLOO/PPO 阶段需要模型自己生成回答，然后 reward model 打分。如果训练数据里已经带了标准答案，RL 就变成 SFT 了。

最后这段：

```python
source_txt = dialog[2 * i]
new_examples["prompt"].append(source_txt)
```

只取每轮的用户输入部分作为 prompt。

### dataset.map

#### 源码

```python
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=training_args.dataset_num_proc,
    remove_columns=train_dataset.column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset" if is_main_process else None,
)
train_dataset = tokenized_train_dataset.filter(
    lambda x: len(x['prompt']) > 0
)
```

#### 逐段解释

这里得到的数据只有一列：

```text
prompt
```

后续 `RLOOTrainer` 会基于 prompt 生成 completion。

### RLOOTrainer 初始化

#### 源码

```python
trainer = RLOOTrainer(
    args=training_args,
    processing_class=tokenizer,
    model=policy,
    reward_funcs=reward_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)
```

#### 逐段解释

`RLOOTrainer` 接收的关键组件：

- `model=policy`：要更新的模型。
- `reward_funcs=reward_model`：奖励函数，这里是 reward model。
- `processing_class=tokenizer`：负责 tokenize/decode。
- `train_dataset`：只含 prompt。
- `peft_config`：如果使用 LoRA，只更新 adapter。

这里的 `reward_funcs` 可以是模型，也可以在某些 TRL 接口里是 Python 函数。MedicalGPT 这份代码选择了 reward model 路线。

### 训练和生成样例

#### 源码

```python
if training_args.do_train:
    if is_main_process:
        logger.info("*** Train ***")
    trainer.train()

    if is_main_process:
        trainer.save_model(training_args.output_dir)

trainer.generate_completions()
```

#### 逐段解释

`trainer.train()` 会执行 RL 更新。训练后保存模型。

`trainer.generate_completions()` 会生成一些样例回答，方便看 RL 后模型输出是否有明显变化。

## 和当前 Qwen3 医疗项目的关系

你项目描述里 PPO 阶段的目标是：

```text
C-Eval 医疗准确率 0.8652 -> 0.8717
复杂病例格式回答准确率 72% -> 94%
```

如果使用当前 `ppo_training.py`，你需要先准备：

1. SFT 后 policy model。
2. 医疗 reward model。
3. 复杂病例 prompt 数据。

其中 reward model 是关键。没有 reward model，这个脚本不能表达“答案医学准确”或“格式合规”。

## 运行命令

示例：

```bash
CUDA_VISIBLE_DEVICES=0 python training/ppo_training.py \
  --sft_model_path outputs/qwen3_4b_medical_qlora_381k_then_top100k \
  --reward_model_path outputs/qwen3_4b_medical_reward_model \
  --train_file_dir data/rl_medical_prompts \
  --validation_file_dir data/rl_medical_prompts \
  --template_name qwen3 \
  --output_dir outputs/qwen3_4b_medical_rloo \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-6 \
  --num_train_epochs 1 \
  --report_to swanlab
```

具体参数是否完全可用，要以当前 TRL `RLOOConfig` 支持项为准。

## 常见坑

- 文件名叫 PPO，但源码实际使用 `RLOOTrainer`。
- 需要 reward model，不是只有 SFT 模型就能跑。
- RL 数据只需要 prompt，不需要标准答案。
- reward model 如果偏置，会把 policy 带偏。
- RL 比 SFT 更容易训飞，学习率要更小。
- 先做 DPO 对照，再做 RLOO/PPO，实验更稳。

## 学习检查清单

- 能解释 policy model 和 reward model 的区别。
- 能说明为什么 RLOO 只从数据里取 prompt。
- 能指出 MedicalGPT 这里不是经典 PPO，而是 RLOO。
- 能说出做医疗 PPO/RLOO 前还缺 reward model。
- 能理解 KL 约束的作用：防止模型远离 SFT 初始行为。
