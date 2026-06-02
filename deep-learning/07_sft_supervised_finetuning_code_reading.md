# MedicalGPT 训练源码精读 07：supervised_finetuning.py

## 整体作用

`training/supervised_finetuning.py` 是 MedicalGPT 的 SFT 主入口。你的 Qwen3 医疗 LoRA / QLoRA 训练就是跑这个文件。

它做的事情可以拆成：

```text
解析参数
  -> 加载 tokenizer
  -> 加载本地 ShareGPT JSONL
  -> conversations 转 prompt/answer
  -> tokenizer 编码
  -> 构造 input_ids / labels
  -> 加载 Qwen3
  -> 配置 LoRA / QLoRA
  -> Trainer 训练
  -> 保存 adapter
```

## 关键参数类

### `ModelArguments`

```python
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default="float16")
    device_map: Optional[str] = field(default="auto")
    trust_remote_code: bool = field(default=True)

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")
```

解释：

- `model_name_or_path`：底座模型，例如 `Qwen/Qwen3-4B-Instruct`。
- `load_in_4bit`：QLoRA 必备。
- `torch_dtype`：一般用 `bfloat16` 或 `float16`。
- `trust_remote_code=True`：Qwen 系列经常需要。

### `DataArguments`

```python
@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None)
    train_file_dir: Optional[str] = field(default=None)
    validation_file_dir: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    validation_split_percentage: Optional[int] = field(default=1)
    preprocessing_num_workers: Optional[int] = field(default=None)
```

解释：

- `train_file_dir` 是你最关心的参数。
- 它会递归读取目录下所有 `.jsonl`。
- 你的主实验目录是：

```text
data/sft_medsft_top100k
```

### `ScriptArguments`

```python
@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True)
    train_on_inputs: bool = field(default=False)
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    peft_path: Optional[str] = field(default=None)
    qlora: bool = field(default=False)
    model_max_length: int = field(default=512)
    template_name: Optional[str] = field(default=None)
```

解释：

- `use_peft=True`：启用 LoRA。
- `qlora=True`：启用 QLoRA 训练逻辑。
- `target_modules=all`：自动找线性层。
- `template_name=qwen3`：让 MedicalGPT 用 Qwen3 prompt 模板。
- `train_on_inputs=False`：只让模型学习 assistant 回答，不学习 user prompt。

## LoRA 保存类

```python
class SavePeftModelTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)
```

解释：

普通 Trainer 保存完整模型；LoRA 训练时只需要保存 adapter。这个类重写 `save_model()`，保存 PEFT 模型权重。

## 自动寻找 LoRA 注入层

```python
def find_all_linear_names(peft_model, int4=False, int8=False):
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)
```

逐段解释：

- 普通 LoRA 找 `torch.nn.Linear`。
- QLoRA 4bit 找 `bnb.nn.Linear4bit`。
- 跳过 `lm_head` 和 `output_layer`，避免直接改输出头。
- 返回模块名集合，例如 `q_proj`、`k_proj`、`v_proj`、`o_proj`。

## 数据加载核心

```python
if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
    train_data_files = glob(f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
    logger.info(f"train files: {train_data_files}")
    data_files["train"] = train_data_files
```

解释：

`train_file_dir` 是目录，不是单个文件。脚本会递归找所有 JSONL。这就是为什么我们把数据放成：

```text
MedicalGPT/data/sft_medsft_top100k/train.jsonl
```

训练时传：

```bash
--train_file_dir data/sft_medsft_top100k
```

## prompt 构造核心

SFT 的关键是把 `conversations` 转成 prompt + answer。

```python
for i, source in enumerate(examples['conversations']):
    messages = []
    for sentence in source:
        role = sentence.get("from", "")
        value = sentence.get("value", "")
        if role in ["human", "user", "observation"]:
            messages.append({"role": "user", "content": value})
        elif role in ["gpt", "assistant", "function_call"]:
            messages.append({"role": "assistant", "content": value})
```

解释：

- MedicalGPT 支持 `human/gpt`，也兼容 `user/assistant`。
- 它先把 ShareGPT 风格统一成 chat message。
- 后续再用 `template_name` 或 tokenizer chat template 拼 prompt。

## labels mask 核心

```python
if script_args.train_on_inputs:
    labels += source_ids + target_ids + [tokenizer.eos_token_id]
else:
    labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]
```

解释：

- `source_ids` 是用户 prompt。
- `target_ids` 是助手回答。
- `IGNORE_INDEX` 表示这部分不算 loss。
- 默认 `train_on_inputs=False`，所以用户输入不参与 loss。

为什么？

SFT 目标是让模型学会生成助手回答，而不是让模型背用户问题。

## 当前 Qwen3 医疗训练命令

```bash
cd MedicalGPT

CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --train_file_dir data/sft_medsft_top100k \
  --validation_file_dir data/sft_medsft_top100k \
  --do_train \
  --do_eval \
  --use_peft True \
  --qlora True \
  --load_in_4bit True \
  --template_name qwen3 \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --model_max_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --torch_dtype bfloat16 \
  --bf16 \
  --optim paged_adamw_32bit \
  --output_dir outputs/qwen3_4b_medical_qlora_top100k
```

## 常见坑

- 数据目录要传目录，不是文件。
- Qwen3 要用 `--template_name qwen3`。
- QLoRA 需要 CUDA + bitsandbytes。
- 显存不够先降 `model_max_length`。
- 如果 loss 为 nan，先降学习率或检查数据。

