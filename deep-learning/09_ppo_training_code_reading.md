# MedicalGPT 训练源码精读 09：ppo_training.py

## 整体作用

`ppo_training.py` 用来做强化学习式对齐。MedicalGPT 这里使用的是 TRL 中的 RLOO / PPO 风格训练。

PPO 类训练通常需要：

- policy model：当前要优化的模型
- reference model：参考模型，用于 KL 约束
- reward model：奖励模型，给生成回答打分
- prompt dataset：用户问题集合

## 参数类：`RLOOArguments`

```python
class RLOOArguments:
    sft_model_path: Optional[str] = field(default=None, metadata={"help": "Path to the SFT model."})
    reward_model_path: Optional[str] = field(default=None, metadata={"help": "Path to the reward model."})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field(default=None, metadata={"help": "Dataset configuration name."})
    dataset_train_split: str = field(default="train")
    dataset_test_split: str = field(default="test")
    train_file_dir: Optional[str] = field(default=None)
    validation_file_dir: Optional[str] = field(default=None)
    template_name: Optional[str] = field(default=None)
    max_source_length: Optional[int] = field(default=1024)
```

解释：

- `sft_model_path`：SFT 后模型，作为 PPO 初始 policy。
- `reward_model_path`：奖励模型路径。
- `train_file_dir`：本地 prompt 数据。
- `template_name`：prompt 模板，例如 `qwen3`。

## 设备和多卡逻辑

```python
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
is_main_process = local_rank == 0
world_size = int(os.environ.get("WORLD_SIZE", "1"))
ddp = world_size != 1
num_gpus = torch.cuda.device_count()
```

解释：

- `LOCAL_RANK`：当前进程对应哪张卡。
- `WORLD_SIZE`：总进程数。
- `ddp`：是否分布式训练。
- PPO 显存压力大，所以多卡管理很重要。

## 加载 reward model

```python
reward_model = AutoModelForSequenceClassification.from_pretrained(
    args.reward_model_path, **reward_model_kwargs
)
```

解释：

奖励模型是分类/打分模型，输出一个标量 reward。PPO 会根据 reward 调整 policy。

## 加载 policy model

```python
sft_model_path = args.sft_model_path or model_args.model_name_or_path
policy = AutoModelForCausalLM.from_pretrained(
    sft_model_path, **policy_kwargs
)
```

解释：

policy 是要被优化的模型。通常从 SFT 模型开始，而不是从 base model 开始。

## 数据加载逻辑

```python
if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
    train_data_files = glob(f'{args.train_file_dir}/**/*.jsonl', recursive=True)
    data_files["train"] = train_data_files
if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
    eval_data_files = glob(f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
    data_files["validation"] = eval_data_files
dataset = load_dataset('json', data_files=data_files)
```

解释：

和 SFT 类似，PPO 也可以读取本地 JSONL 目录。

## prompt 预处理

```python
def preprocess_function(examples):
    new_examples = {"prompt": []}

    def get_dialog(examples):
        for i, source in enumerate(examples['conversations']):
            ...
            if prompt_template:
                yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)
            else:
                ...

    for dialog in get_dialog(examples):
        for i in range(len(dialog) // 2):
            source_txt = dialog[2 * i]
            new_examples["prompt"].append(source_txt)

    return new_examples
```

解释：

PPO 不需要固定答案，它需要 prompt，让 policy 自己生成回答，然后 reward model 打分。

所以输出字段是：

```text
prompt
```

而不是 SFT 里的 `input_ids/labels`。

## PPO 和 DPO 的区别

DPO：

```text
离线数据：prompt + chosen + rejected
不需要 reward model
不需要在线生成
```

PPO：

```text
在线生成回答
reward model 打分
根据 reward 更新 policy
```

## 和当前项目的关系

你项目里写到“复杂病例格式回答准确率 72% -> 94%”，这种目标更接近 PPO / GRPO：

- 先让模型回答复杂病例。
- 检查格式是否合规。
- 检查医学答案是否准确。
- 用奖励函数或奖励模型优化。

## 常见坑

- PPO 比 SFT/DPO 更难稳定。
- reward model 质量决定训练质量。
- 显存需求更高。
- reward hacking 风险更高。

