# MedicalGPT 训练源码精读 07：supervised_finetuning.py

## 整体作用

`MedicalGPT/training/supervised_finetuning.py` 是 MedicalGPT 的 SFT 入口。它做的事情可以概括成一句话：

把 ShareGPT 格式的 `conversations` 数据读进来，套上 Qwen3 对话模板，tokenize 成 `input_ids / attention_mask / labels`，再用 HuggingFace `Trainer` 对 Qwen3-4B-Instruct 做 LoRA 或 QLoRA 微调。

在当前项目里，它对应的是这条训练链路：

```text
cleaned_alpaca.jsonl
  -> ShareGPT train.jsonl
  -> supervised_finetuning.py
  -> Qwen3-4B-Instruct + LoRA / QLoRA adapter
```

## 方法原理

SFT，全称 Supervised Fine-Tuning，监督微调。对自回归语言模型来说，本质还是“预测下一个 token”：

```text
输入：用户问题 + 已生成的前文 token
目标：预测 assistant 回复里的下一个 token
损失：Cross Entropy Loss
```

训练时使用 teacher forcing：模型不是真的一步一步自由生成，而是把标准答案也喂进去，让模型在每个位置预测标准答案的下一个 token。这样速度快、梯度稳定。

关键点是：用户输入不应该参与 loss。我们希望模型学习“怎么回答”，而不是学习“怎么复读用户问题”。所以源码里会把用户 prompt 对应的 label 设置成 `IGNORE_INDEX`，让 loss 函数忽略这些位置。

最关键的一行是：

```python
labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]
```

它的含义是：

- `source_ids`：用户 prompt、系统提示、模板符号，不参与训练损失。
- `target_ids`：assistant 的答案，参与训练损失。
- `eos_token_id`：回答结束符，也参与训练，让模型学会停止。

## 脚本输入输出

输入：

```text
--model_name_or_path Qwen/Qwen3-4B-Instruct
--train_file_dir data/sft_medsft_cleaned_381k
--validation_file_dir data/sft_medsft_cleaned_381k
--template_name qwen3
```

训练数据格式：

```json
{"conversations": [{"from": "human", "value": "患者发热怎么办？"}, {"from": "gpt", "value": "建议结合体温、症状和基础疾病判断..."}]}
```

输出：

```text
outputs/qwen3_4b_medical_qlora_381k/
  adapter_config.json
  adapter_model.safetensors
  tokenizer files
  trainer_state.json
  train_results.json
  eval_results.json
```

如果是 LoRA / QLoRA，输出主要是 adapter，不是完整 4B 权重。

## 主流程图

```text
解析参数
  -> 加载 tokenizer
  -> 加载 ShareGPT JSONL 数据
  -> 根据 template_name 构造 prompt
  -> tokenize prompt 和 answer
  -> mask 用户输入 labels
  -> 加载 Qwen3 模型
  -> 配置 4bit / LoRA
  -> Trainer 训练
  -> 保存 adapter 和 tokenizer
```

## 关键源码精读

### 参数类：模型、数据和脚本参数

#### 源码

```python
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
    tokenizer_name_or_path: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: Optional[str] = field(default="main")
    hf_hub_token: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=False)
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={"choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    device_map: Optional[str] = field(default="auto")
    trust_remote_code: bool = field(default=True)
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(default=None)
    flash_attn: Optional[bool] = field(default=False)
    shift_attn: Optional[bool] = field(default=False)
    neft_alpha: Optional[float] = field(default=0)

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file_dir: Optional[str] = field(default=None)
    validation_file_dir: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    ignore_pad_token_for_loss: bool = field(default=True)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=1)
    preprocessing_num_workers: Optional[int] = field(default=None)


@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True)
    train_on_inputs: bool = field(default=False)
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    qlora: bool = field(default=False)
    model_max_length: int = field(default=512)
    template_name: Optional[str] = field(default=None)
    tool_format: Optional[str] = field(default=None)
```

#### 逐段解释

`ModelArguments` 管模型从哪里来、怎么加载。当前你训练 Qwen3 时，最重要的是：

- `model_name_or_path`：可以是 HuggingFace 模型名，例如 `Qwen/Qwen3-4B-Instruct`，也可以是本地目录。
- `load_in_4bit`：是否 4bit 量化加载。QLoRA 通常要打开。
- `torch_dtype`：推荐在新显卡上用 `bfloat16`。
- `device_map`：默认 `auto`，单卡会放到 cuda，部分多卡会自动切层。

`DataArguments` 管数据从哪里来。当前项目使用本地 JSONL，所以主要用：

- `train_file_dir`
- `validation_file_dir`
- `max_train_samples`
- `max_eval_samples`

注意这里要传目录，不是单个文件。MedicalGPT 会递归找目录下面所有 `*.jsonl`。

`ScriptArguments` 管训练方式：

- `use_peft=True` 表示走 LoRA。
- `qlora=True` 表示 4bit base model + LoRA adapter。
- `template_name=qwen3` 表示使用 Qwen3 对话模板。
- `train_on_inputs=False` 是重点，表示用户输入不参与 loss。

#### 为什么这样写

MedicalGPT 把参数拆成三组，是为了让训练脚本同时支持不同模型、不同数据源、不同微调方式。你当前最容易混淆的是 `--template_name qwen3` 和 `--model_name_or_path`：

- `--template_name qwen3` 只是 prompt 模板。
- `--model_name_or_path Qwen/Qwen3-4B-Instruct` 才是真正加载 Qwen3 模型。

### 保存 LoRA 的 Trainer

#### 源码

```python
class SavePeftModelTrainer(Trainer):
    """
    Trainer for lora models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)
```

#### 逐段解释

这个类继承 HuggingFace `Trainer`，只改了保存逻辑。普通 `Trainer.save_model()` 可能会保存完整模型；PEFT 场景下，我们只需要保存 LoRA adapter。

`self.model.save_pretrained(output_dir)` 对 PEFT 模型来说会保存：

```text
adapter_config.json
adapter_model.safetensors
```

#### 为什么这样写

Qwen3-4B 完整模型很大，不适合每次 checkpoint 都保存一份。LoRA 只保存增量权重，通常几十 MB 到几百 MB，更适合实验迭代。

### 自动寻找 LoRA 注入层

#### 源码

```python
def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
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
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)
```

#### 逐段解释

第一段决定要找哪类线性层：

- 普通 LoRA 找 `torch.nn.Linear`。
- 4bit QLoRA 找 `bnb.nn.Linear4bit`。
- 8bit 找 `bnb.nn.Linear8bitLt`。

第二段遍历模型的所有子模块：

```python
for name, module in peft_model.named_modules():
```

它会看到类似：

```text
model.layers.0.self_attn.q_proj
model.layers.0.self_attn.k_proj
model.layers.0.mlp.gate_proj
```

第三段跳过 `lm_head` 和 `output_layer`。这些是最终输出词表 logits 的层，通常不注入 LoRA，避免影响输出头稳定性。

最后只保留最后一级名字：

```python
names = name.split('.')
lora_module_names.add(names[0] if len(names) == 1 else names[-1])
```

对 Qwen3 来说，常见结果会包含：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

#### 为什么这样写

你命令里写 `--target_modules all` 时，源码会自动调用这个函数。好处是不用手动猜 Qwen3 的层名；坏处是可能注入层较多，显存更紧。如果 OOM，可以手动指定：

```text
--target_modules q_proj,k_proj,v_proj,o_proj
```

### tokenizer 加载和特殊 token

#### 源码

```python
tokenizer_kwargs = {
    "cache_dir": model_args.cache_dir,
    "use_fast": model_args.use_fast_tokenizer,
    "trust_remote_code": model_args.trust_remote_code,
}
tokenizer_name_or_path = model_args.tokenizer_name_or_path
if not tokenizer_name_or_path:
    tokenizer_name_or_path = model_args.model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
prompt_template = None
if script_args.template_name:
    prompt_template = get_conv_template(script_args.template_name)
if tokenizer.eos_token_id is None:
    if prompt_template:
        tokenizer.eos_token = prompt_template.stop_str
    else:
        tokenizer.eos_token = "</s>"
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

如果没指定 `tokenizer_name_or_path`，就默认和模型路径一致。你用 HF 直接拉 Qwen3 时，就是：

```text
Qwen/Qwen3-4B-Instruct
```

`get_conv_template(script_args.template_name)` 会从 MedicalGPT 的模板系统里拿到 `qwen3` 的对话格式。模板负责把 ShareGPT 的 human/gpt 变成 Qwen3 能理解的聊天文本。

后面三段是在补齐 `eos / bos / pad`。训练时必须有 `pad_token_id`，因为 batch 内不同样本长度不一样，需要 padding。

#### 为什么这样写

不同开源模型 tokenizer 的特殊 token 不完全一致。这个脚本做了兜底，避免因为缺 `pad_token` 训练直接崩掉。

### 数据加载：读取目录下 JSONL

#### 源码

```python
data_files = {}
if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
    train_data_files = glob(f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
    logger.info(f"train files: {train_data_files}")
    data_files["train"] = train_data_files
if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
    eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
    logger.info(f"eval files: {eval_data_files}")
    data_files["validation"] = eval_data_files
raw_datasets = load_local_json_datasets(data_files, cache_dir=model_args.cache_dir)
if "validation" not in raw_datasets.keys():
    shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
    split = shuffled_train_dataset.train_test_split(
        test_size=float(data_args.validation_split_percentage / 100),
        seed=42
    )
    raw_datasets["train"] = split["train"]
    raw_datasets["validation"] = split["test"]
```

#### 逐段解释

`glob(.../**/*.jsonl, recursive=True)` 表示目录下所有 JSONL 都会被加载。所以你的目录结构应该是：

```text
data/sft_medsft_cleaned_381k/
  train.jsonl
```

而不是把文件直接传给 `--train_file_dir`。

如果你没有传验证集目录，源码会从训练集切 1% 作为 validation。你现在为了方便，把 `validation_file_dir` 也指向同一个目录，再配 `--max_eval_samples 1000`，能快速评估 loss。

#### 为什么这样写

训练大数据时，一个目录里可能有多个 shard，例如：

```text
train-00000.jsonl
train-00001.jsonl
```

递归加载目录比指定单文件更灵活。

### prompt 构造：从 ShareGPT conversations 到多轮对话

#### 源码

```python
def preprocess_function(examples):
    input_ids_list = []
    attention_mask_list = []
    targets_list = []
    roles = ["human", "gpt"]

    def get_dialog(examples):
        system_prompts = examples.get("system_prompt", "")
        for i, source in enumerate(examples['conversations']):
            system_prompt = ""
            tools_text = ""
            if "tools" in examples and examples["tools"][i]:
                tools_json = examples["tools"][i]
                if isinstance(tools_json, str):
                    tools_parsed = json.loads(tools_json)
                    if tools_parsed and script_args.tool_format:
                        tu = get_tool_utils(script_args.tool_format)
                        tools_text = tu.tool_formatter(tools_parsed)

            messages = []
            for sentence in source:
                role = sentence.get("from", "")
                value = sentence.get("value", "")

                if role == "system":
                    system_prompt = value
                    continue

                if role in ["human", "user", "observation"]:
                    messages.append({"role": "user", "content": value})
                elif role in ["gpt", "assistant", "function_call"]:
                    messages.append({"role": "assistant", "content": value})

            if tools_text:
                system_prompt = system_prompt + ("\n\n" if system_prompt else "") + tools_text

            history_messages = []
            temp_history = []
            for msg in messages:
                if not temp_history and msg["role"] == "user":
                    temp_history.append(msg["content"])
                elif len(temp_history) == 1 and msg["role"] == "assistant":
                    temp_history.append(msg["content"])
                    history_messages.append(temp_history)
                    temp_history = []
                elif msg["role"] == "user" and len(temp_history) == 1:
                    temp_history[0] += "\n" + msg["content"]
                elif msg["role"] == "assistant" and len(temp_history) == 0:
                    pass
                elif msg["role"] == "assistant" and len(temp_history) == 2:
                    history_messages[-1][1] += "\n" + msg["content"]

            if not history_messages:
                continue

            if not system_prompt:
                system_prompt = system_prompts[i] if system_prompts else ""
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
```

#### 逐段解释

`preprocess_function()` 是 dataset `map()` 调用的函数。它一次处理一批样本。

`get_dialog()` 内部先读取 `examples['conversations']`。每个 `source` 是一条 ShareGPT 多轮对话：

```json
[
  {"from": "human", "value": "问题"},
  {"from": "gpt", "value": "回答"}
]
```

源码把不同角色统一成标准角色：

- `human` / `user` -> `user`
- `gpt` / `assistant` -> `assistant`
- `system` -> 系统提示

`history_messages` 最后会变成：

```python
[
    ["用户第一轮问题", "助手第一轮回答"],
    ["用户第二轮问题", "助手第二轮回答"],
]
```

如果指定了 `--template_name qwen3`，就走：

```python
prompt_template.get_dialog(history_messages, system_prompt=system_prompt)
```

否则走 tokenizer 自带的：

```python
tokenizer.apply_chat_template(...)
```

#### 为什么这样写

训练数据格式是 ShareGPT，但模型实际吃的是 token ids。中间必须经过“模板渲染”这一步。你不需要手写 Qwen3 特殊 token，因为 `template_name=qwen3` 会处理。

### tokenize 和 labels mask

#### 源码

```python
for dialog in get_dialog(examples):
    input_ids, labels = [], []

    for i in range(len(dialog) // 2):
        source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0))
        target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

        total_len = len(source_ids) + len(target_ids)
        max_source_len = int(max_length * (len(source_ids) / total_len))
        max_target_len = int(max_length * (len(target_ids) / total_len))

        if len(source_ids) > max_source_len:
            source_ids = source_ids[:max_source_len]
        if len(target_ids) > max_target_len - 1:
            target_ids = target_ids[:max_target_len - 1]
        if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
            source_ids = source_ids[1:]
        if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
            target_ids = target_ids[:-1]
        if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
            break

        input_ids += source_ids + target_ids + [tokenizer.eos_token_id]
        if script_args.train_on_inputs:
            labels += source_ids + target_ids + [tokenizer.eos_token_id]
        else:
            labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

    input_ids_list.append(input_ids)
    attention_mask_list.append([1] * len(input_ids))
    targets_list.append(labels)
```

#### 逐段解释

`dialog` 是交替排列的列表：

```text
[用户prompt片段, assistant答案, 用户prompt片段, assistant答案]
```

每轮分别 tokenize：

```python
source_ids = tokenizer.encode(dialog[2 * i])
target_ids = tokenizer.encode(dialog[2 * i + 1])
```

源码会按比例截断 prompt 和 answer，保证总长度不超过 `model_max_length`。这对 4B 模型非常关键，因为显存和序列长度近似线性甚至更高关系增长。

最重要的是 label 构造：

```python
labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]
```

`IGNORE_INDEX` 来自：

```python
IGNORE_INDEX = LabelSmoother.ignore_index
```

通常是 `-100`。PyTorch 交叉熵遇到 `-100` 会跳过该位置，不计算 loss。

#### 为什么这里只训练 assistant 回复

假设原文是：

```text
用户：什么是高血压？
助手：高血压是指...
```

我们希望模型学会在看到“什么是高血压？”后回答“高血压是指...”。如果用户问题也参与 loss，模型会学“复现用户问题”，这会浪费训练信号，还可能让模型更爱重复问题。

所以当前医疗 SFT 应该保持：

```text
--train_on_inputs False
```

### dataset.map 进入 tokenization

#### 源码

```python
train_dataset = raw_datasets['train'].shuffle(seed=42)
max_train_samples = len(train_dataset)
if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
    train_dataset = train_dataset.select(range(max_train_samples))

with training_args.main_process_first(desc="Train dataset tokenization"):
    tokenized_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset" if is_main_process else None,
    )
    train_dataset = tokenized_dataset.filter(
        filter_empty_labels,
        num_proc=data_args.preprocessing_num_workers
    )
```

#### 逐段解释

`shuffle(seed=42)` 保证训练顺序可复现。

`max_train_samples` 是 smoke test 的关键参数。如果你先跑 200 条，可以传：

```text
--max_train_samples 200
```

`map(preprocess_function, batched=True)` 会把原始 `conversations` 转成模型训练字段：

```text
input_ids
attention_mask
labels
```

`remove_columns=train_dataset.column_names` 会删掉原始文本字段，避免 Trainer 看到多余列。

`filter_empty_labels` 会过滤掉 labels 全是 `IGNORE_INDEX` 的样本。也就是说，如果某条数据没有 assistant 回答，它不会进入训练。

### 模型加载和 QLoRA 量化

#### 源码

```python
load_in_4bit = model_args.load_in_4bit
load_in_8bit = model_args.load_in_8bit
quantization_config = None
if load_in_4bit and load_in_8bit:
    raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")
elif load_in_8bit or load_in_4bit:
    logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
    if is_deepspeed_zero3_enabled():
        raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif load_in_4bit:
        if script_args.qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )

model_kwargs = {
    "config": config,
    "torch_dtype": torch_dtype,
    "trust_remote_code": model_args.trust_remote_code,
    "quantization_config": quantization_config,
    "low_cpu_mem_usage": True,
    "device_map": model_args.device_map,
}

model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    **model_kwargs
)
```

#### 逐段解释

`load_in_4bit` 和 `load_in_8bit` 不能同时开。

QLoRA 的核心配置是：

```python
bnb_4bit_quant_type="nf4"
bnb_4bit_use_double_quant=True
```

NF4 是 QLoRA 论文常用的 4bit 量化类型，适合正态分布权重。double quant 会进一步压缩量化常数，减少显存。

`AutoModelForCausalLM.from_pretrained()` 才是真正加载 Qwen3 的地方。你如果没有本地模型，但服务器能访问 HF 或 mirror，可以直接传：

```text
--model_name_or_path Qwen/Qwen3-4B-Instruct
```

#### 为什么 QLoRA 适合你

Qwen3-4B 全参微调显存压力大。QLoRA 把 base model 4bit 冻住，只训练 LoRA 小矩阵，适合租单卡做实验。

### 配置 LoRA adapter

#### 源码

```python
if script_args.use_peft:
    logger.info("Fine-tuning method: LoRA(PEFT)")

    if script_args.peft_path is not None:
        logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
        model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
    else:
        logger.info("Init new peft model")
        if load_in_8bit or load_in_4bit:
            model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)
        target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
        if target_modules and 'all' in target_modules:
            target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)
        modules_to_save = script_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=script_args.lora_rank,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            modules_to_save=modules_to_save)
        model = get_peft_model(model, peft_config)
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)
    model.print_trainable_parameters()
```

#### 逐段解释

如果传了 `--peft_path`，脚本会加载已有 adapter 继续训练。这正好对应你的两阶段训练：

```text
381k SFT adapter -> 继续用 top100k SFT
```

如果没传 `--peft_path`，就新建 LoRA。

`prepare_model_for_kbit_training()` 是 QLoRA 必需步骤之一，它会处理量化模型训练时的 dtype、梯度检查点等细节。

`LoraConfig` 里：

- `r` 是 LoRA 秩，越大可训练参数越多。
- `lora_alpha` 是缩放系数。
- `lora_dropout` 是 LoRA 分支 dropout。
- `target_modules` 是注入哪些线性层。

#### 为什么可训练参数转 float32

```python
for param in filter(lambda p: p.requires_grad, model.parameters()):
    param.data = param.data.to(torch.float32)
```

这行把可训练 LoRA 参数转成 fp32，有助于避免 fp16 梯度不稳定。

### DataCollator 和 Trainer

#### 源码

```python
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=IGNORE_INDEX,
    pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,
)

trainer = SavePeftModelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    processing_class=tokenizer,
    data_collator=data_collator,
)
```

#### 逐段解释

`DataCollatorForSeq2Seq` 负责把不同长度的样本 padding 到同一长度。

`label_pad_token_id=IGNORE_INDEX` 表示 labels padding 的部分也不参与 loss。

`SavePeftModelTrainer` 是前面自定义的 Trainer，保存时更适配 LoRA。

### 训练、评估和保存

#### 源码

```python
if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = max_train_samples
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    model.config.use_cache = True
    tokenizer.padding_side = "left"
    tokenizer.init_kwargs["padding_side"] = "left"

    if trainer.is_world_process_zero():
        if is_deepspeed_zero3_enabled():
            save_model_zero3(model, tokenizer, training_args, trainer)
        else:
            save_model(model, tokenizer, training_args)

if training_args.do_eval:
    metrics = trainer.evaluate(metric_key_prefix="eval")
    metrics["eval_samples"] = max_eval_samples
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
```

#### 逐段解释

`trainer.train()` 是真正训练入口。训练完成后会记录：

- train loss
- learning rate
- epoch
- runtime
- samples per second

评估时：

```python
perplexity = math.exp(metrics["eval_loss"])
```

PPL 越低，说明模型对验证文本越“熟悉”。你项目目标里提到 1K 医疗长文本 PPL 从 `15.194` 降到 `9.823`，就是类似逻辑。

## 和当前 Qwen3 医疗项目的关系

你现在最推荐的 SFT 路线是：

1. 用清洗后 381621 条训练第一阶段，得到通用医疗 SFT adapter。
2. 用 C-Eval 相似筛选的 100000 条继续训练，得到更贴近医学考试目标域的 adapter。

第一阶段：

```bash
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --train_file_dir data/sft_medsft_cleaned_381k \
  --validation_file_dir data/sft_medsft_cleaned_381k \
  --do_train \
  --do_eval \
  --use_peft True \
  --qlora True \
  --load_in_4bit True \
  --max_train_samples -1 \
  --max_eval_samples 1000 \
  --model_max_length 2048 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --output_dir outputs/qwen3_4b_medical_qlora_381k \
  --template_name qwen3 \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to swanlab \
  --run_name qwen3-4b-medical-qlora-381k
```

第二阶段：

```bash
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --peft_path outputs/qwen3_4b_medical_qlora_381k \
  --train_file_dir data/sft_medsft_top100k \
  --validation_file_dir data/sft_medsft_top100k \
  --do_train \
  --do_eval \
  --use_peft True \
  --qlora True \
  --load_in_4bit True \
  --max_train_samples -1 \
  --max_eval_samples 1000 \
  --model_max_length 2048 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --output_dir outputs/qwen3_4b_medical_qlora_381k_then_top100k \
  --template_name qwen3 \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to swanlab \
  --run_name qwen3-4b-medical-qlora-381k-then-top100k
```

## 常见坑

- `--template_name qwen3` 不是模型路径，它只是聊天模板。
- `--train_file_dir` 要传目录，不要传 JSONL 文件。
- Alpaca JSONL 不能直接喂这个脚本，先转成 ShareGPT。
- QLoRA 时不要同时开 DeepSpeed ZeRO-3。
- OOM 时先把 `model_max_length` 从 2048 降到 1024。
- 如果 `target_modules all` OOM，改成 `q_proj,k_proj,v_proj,o_proj`。
- `--max_eval_samples` 不要太大，评估会拖慢训练。

## 学习检查清单

- 能说清楚 SFT 的 loss 是怎么来的。
- 能解释为什么用户 prompt 的 label 是 `IGNORE_INDEX`。
- 能看懂 `conversations` 如何变成 Qwen3 prompt。
- 能说清楚 LoRA 和 QLoRA 的区别。
- 能知道 adapter 保存在哪里，以及为什么不是完整模型。
