# MedicalGPT 训练源码精读 08：dpo_training.py

## 整体作用

`MedicalGPT/training/dpo_training.py` 是偏好对齐训练入口。它读入 preference 数据，把同一个 prompt 下的好答案 `chosen` 和差答案 `rejected` 送进 TRL 的 `DPOTrainer`，让模型更倾向于输出 chosen。

它不是继续学习“标准答案长什么样”，而是学习“两个答案谁更好”。

当前项目里，DPO 可以作为 PPO / GRPO 前的稳定对照实验：

```text
复杂病例 prompt
  -> chosen 高质量医学回答
  -> rejected 低质量或格式错误回答
  -> DPO 训练
  -> 更偏好安全、准确、格式合规的回答
```

## 方法原理

DPO，全称 Direct Preference Optimization。它的核心思想是：不显式训练 reward model，也不跑在线 PPO 采样，而是直接用偏好对训练语言模型。

一条 DPO 数据长这样：

```json
{
  "conversations": [{"from": "human", "value": "患者胸痛怎么办？"}],
  "chosen": "应优先评估生命体征、胸痛性质，并警惕急性冠脉综合征...",
  "rejected": "胸痛不用管，休息一下就行。"
}
```

DPO 会比较：

```text
模型给 chosen 的概率
模型给 rejected 的概率
```

并让 chosen 的相对概率变高。直观 DPO loss 是：

```text
loss = -log sigmoid(beta * (模型偏好差值 - reference 偏好差值))
```

其中：

- `policy model`：正在训练的模型。
- `reference model`：参考模型，通常是 SFT 后的冻结模型。
- `chosen`：更好的答案。
- `rejected`：更差的答案。
- `beta`：控制偏好优化强度，越大越激进。

MedicalGPT 这里使用 TRL 的 `DPOTrainer` 封装了 loss 细节。

## 脚本输入输出

输入数据目录：

```text
data/preference_medical/
  train.jsonl
```

每行至少包含：

```json
{"conversations": [...], "chosen": "...", "rejected": "..."}
```

输出：

```text
outputs/qwen3_4b_medical_dpo/
  adapter_config.json
  adapter_model.safetensors
  trainer_state.json
```

## 主流程图

```text
解析 ScriptArguments
  -> 加载 tokenizer 和 qwen3 template
  -> 加载 preference JSONL
  -> conversations 构造成 prompt
  -> 取 chosen / rejected
  -> 过滤过长样本
  -> 加载 base/SFT 模型
  -> 构造 DPOConfig
  -> 构造 LoRA config
  -> DPOTrainer 训练
  -> 保存 adapter
```

## 关键源码精读

### 参数类：ScriptArguments

#### 源码

```python
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with DPO
    """
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for weights initialization."}
    )
    tokenizer_name_or_path: Optional[str] = field(default=None)
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=False)
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    device_map: Optional[str] = field(default="auto")
    trust_remote_code: bool = field(default=True)

    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file_dir: Optional[str] = field(default=None)
    validation_file_dir: Optional[str] = field(default=None)
    template_name: Optional[str] = field(default=None)
    tool_format: Optional[str] = field(default=None)
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    max_source_length: Optional[int] = field(default=2048)
    max_target_length: Optional[int] = field(default=512)
    min_target_length: Optional[int] = field(default=4)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=1)
    preprocessing_num_workers: Optional[int] = field(default=4)

    use_peft: bool = field(default=True)
    qlora: bool = field(default=False)
    target_modules: Optional[str] = field(default=None)
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=16.0)
    peft_path: Optional[str] = field(default=None)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    learning_rate: Optional[float] = field(default=5e-4)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    warmup_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.05)
    optim: Optional[str] = field(default="adamw_torch")
    fp16: Optional[bool] = field(default=True)
    bf16: Optional[bool] = field(default=False)
    gradient_checkpointing: Optional[bool] = field(default=True)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    save_steps: Optional[int] = field(default=50)
    eval_steps: Optional[int] = field(default=50)
    logging_steps: Optional[int] = field(default=1)
    output_dir: Optional[str] = field(default="outputs-dpo")
    max_steps: Optional[int] = field(default=200)
    eval_strategy: Optional[str] = field(default="steps")
    remove_unused_columns: Optional[bool] = field(default=False)
    report_to: Optional[str] = field(default="tensorboard")

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")
```

#### 逐段解释

这个脚本不像 SFT 那样拆成 `ModelArguments / DataArguments / ScriptArguments`，而是把所有参数放进一个 `ScriptArguments`。

模型相关参数决定加载哪个 SFT 后模型继续对齐：

```text
--model_name_or_path outputs/qwen3_4b_medical_qlora_381k_then_top100k
```

数据相关参数决定 preference 数据从哪里来：

```text
--train_file_dir data/preference_medical
```

训练相关参数里没有直接暴露 `beta`，但 `DPOConfig` 有自己的默认值。后续如果需要细调 beta，需要检查当前 TRL 版本是否支持通过脚本参数传入。

### tokenizer 和模板加载

#### 源码

```python
tokenizer_kwargs = {
    "cache_dir": args.cache_dir,
    "use_fast": args.use_fast_tokenizer,
    "trust_remote_code": args.trust_remote_code,
}
tokenizer_name_or_path = args.tokenizer_name_or_path
if not tokenizer_name_or_path:
    tokenizer_name_or_path = args.model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
prompt_template = None
if args.template_name:
    prompt_template = get_conv_template(args.template_name)
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

这里和 SFT 类似。DPO 的 prompt 也必须和 SFT 的 chat template 对齐，否则模型看到的输入格式会变，偏好训练会不稳定。

当前 Qwen3 项目继续使用：

```text
--template_name qwen3
```

### 数据加载：本地 preference JSONL

#### 源码

```python
data_files = {}
if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
    train_data_files = glob(f'{args.train_file_dir}/**/*.jsonl', recursive=True)
    logger.info(f"train files: {', '.join(train_data_files)}")
    data_files["train"] = train_data_files
if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
    eval_data_files = glob(f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
    logger.info(f"eval files: {', '.join(eval_data_files)}")
    data_files["validation"] = eval_data_files
raw_datasets = load_local_json_datasets(data_files, cache_dir=args.cache_dir)
if "validation" not in raw_datasets.keys():
    train_dataset = raw_datasets["train"]
    validation_size = int(len(train_dataset) * args.validation_split_percentage / 100)
    raw_datasets["validation"] = train_dataset.select(range(validation_size))
    raw_datasets["train"] = train_dataset.select(range(validation_size, len(train_dataset)))
```

#### 逐段解释

DPO 数据同样传目录。目录中所有 `*.jsonl` 会被当作训练数据。

如果没给 validation，脚本会从训练集前面切出一部分作为 validation。注意这里不是 shuffle 后切，而是直接按顺序切：

```python
raw_datasets["validation"] = train_dataset.select(range(validation_size))
```

所以 preference 数据最好先打乱，避免 validation 全是某一类样本。

### 构造 prompt：工具调用和多轮对话

#### 源码

```python
def _format_tool_call_value(value, tool_fmt):
    """Format a function_call value using tool_utils."""
    fc_dict = json.loads(value)
    if "name" in fc_dict and "arguments" in fc_dict:
        if tool_fmt:
            tu = get_tool_utils(tool_fmt)
            return tu.function_formatter(
                [FunctionCall(fc_dict["name"], json.dumps(fc_dict["arguments"], ensure_ascii=False))]
            )
        else:
            return f"Action: {fc_dict['name']}\nAction Input: {json.dumps(fc_dict['arguments'], ensure_ascii=False)}"
    return value


def _format_observation_value(value, tool_fmt):
    """Format an observation value for the tool format."""
    if tool_fmt == "qwen":
        return f"<tool_response>\n{value}\n</tool_response>"
    elif tool_fmt == "glm4":
        return f"<|observation|>\n{value}"
    elif tool_fmt == "mistral":
        return f'[TOOL_RESULTS] {{"content": {value}}}[/TOOL_RESULTS]'
    else:
        return f"Observation: {value}"
```

#### 逐段解释

这两个函数是为 agent/tool 数据准备的。你的医疗问答数据一般不用工具调用，但读源码要知道它们在干什么：

- `function_call`：模型调用工具。
- `observation`：工具返回结果。
- `tool_format`：不同模型的工具调用格式不同，Qwen、GLM、Mistral 都有自己的包装方式。

当前医学 SFT/DPO 主线可以不传 `--tool_format`。

### 核心函数：_build_prompt_from_conversations()

#### 源码

```python
def _build_prompt_from_conversations(source, tools_json, system_prompt_override, tool_fmt):
    """Build a prompt string from sharegpt-style conversations with tool call support."""
    system_prompt = system_prompt_override or ""
    tools_text = ""

    if tools_json:
        tools_parsed = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
        if tools_parsed and tool_fmt:
            tu = get_tool_utils(tool_fmt)
            tools_text = tu.tool_formatter(tools_parsed)

    chat_messages = []
    for sentence in source:
        role = sentence.get("from", "")
        value = sentence.get("value", "")

        if role == "system":
            system_prompt = value
            continue

        if role in ["human", "user", "observation"]:
            if role == "observation":
                value = _format_observation_value(value, tool_fmt)
            chat_messages.append({"role": "user", "content": value})
        elif role in ["gpt", "assistant", "function_call"]:
            if role == "function_call":
                value = _format_tool_call_value(value, tool_fmt)
            chat_messages.append({"role": "assistant", "content": value})

    if tools_text:
        system_prompt = system_prompt + ("\n\n" if system_prompt else "") + tools_text

    history_messages = []
    temp = []
    for msg in chat_messages:
        if not temp and msg["role"] == "user":
            temp.append(msg["content"])
        elif len(temp) == 1 and msg["role"] == "assistant":
            temp.append(msg["content"])
            history_messages.append(temp)
            temp = []
        elif msg["role"] == "user" and len(temp) == 1:
            temp[0] += "\n" + msg["content"]
        elif msg["role"] == "assistant" and len(temp) == 2:
            history_messages[-1][1] += "\n" + msg["content"]
    if len(temp) == 1:
        history_messages.append([temp[0], ""])

    if prompt_template:
        return prompt_template.get_prompt(messages=history_messages, system_prompt=system_prompt)
    else:
        accumulated = []
        if system_prompt:
            accumulated.append({"role": "system", "content": system_prompt})
        for uq, br in history_messages:
            accumulated.append({"role": "user", "content": uq})
            if br:
                accumulated.append({"role": "assistant", "content": br})
        return tokenizer.apply_chat_template(
            accumulated, tokenize=False, add_generation_prompt=True
        )
```

#### 逐段解释

这个函数把 ShareGPT 的 `conversations` 转成一个 prompt 字符串。DPO 不把 chosen/rejected 放进 prompt，prompt 只包含历史对话。

`chat_messages` 先统一角色。然后 `history_messages` 把消息整理成问答对：

```python
[
    ["用户问题", "历史助手回答"],
    ["当前用户问题", ""]
]
```

最后如果有 `prompt_template`，用 MedicalGPT 模板；否则用 tokenizer 自带 chat template。

#### 关键点

DPO 的 prompt 必须是 chosen 和 rejected 的共同前缀。不能把 chosen 先塞进 prompt，否则模型就不是在比较两个答案，而是在复读。

### 构造 DPO 数据：return_prompt_and_responses()

#### 源码

```python
def return_prompt_and_responses(examples) -> Dict[str, str]:
    """Load the paired dataset and convert it to the necessary format.

    Data format (ShareGPT):
        {conversations: [{from, value}, ...], chosen: str|dict, rejected: str|dict, tools?: str}

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }
    """
    prompts = []
    chosen_list = []
    rejected_list = []

    tools_list = examples.get("tools", [None] * len(examples["conversations"]))
    for i, source in enumerate(examples["conversations"]):
        tools_json = tools_list[i] if tools_list else None

        chosen_text = examples["chosen"][i]
        rejected_text = examples["rejected"][i]

        prompt = _build_prompt_from_conversations(
            source, tools_json, "", args.tool_format
        )
        prompts.append(prompt)
        chosen_list.append(chosen_text)
        rejected_list.append(rejected_text)

    return {
        "prompt": prompts,
        "chosen": chosen_list,
        "rejected": rejected_list,
    }
```

#### 逐段解释

这个函数是 DPO 数据转换的核心。输入是原始 batch，输出是 TRL `DPOTrainer` 需要的三列：

```text
prompt
chosen
rejected
```

`prompt` 来自 `conversations`，`chosen/rejected` 直接来自数据字段。

#### 为什么 DPO 数据必须这样组织

DPO 学的是同一问题下答案偏好。如果 `chosen` 和 `rejected` 对应的 prompt 不一致，loss 就没有意义。

医疗偏好数据应该保证：

- chosen 医学事实更准确。
- chosen 安全边界更清楚。
- chosen 格式更符合要求。
- rejected 可以是事实错误、格式错误、过度诊断或缺少就医建议。

### 数据 map 和长度过滤

#### 源码

```python
tokenized_dataset = train_dataset.shuffle().map(
    return_prompt_and_responses,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    remove_columns=train_dataset.column_names,
    load_from_cache_file=not args.overwrite_cache,
    desc="Running tokenizer on dataset",
)
train_dataset = tokenized_dataset.filter(
    lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
              and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
)
```

#### 逐段解释

先用 `map()` 生成 `prompt/chosen/rejected`。

再用 `filter()` 丢掉过长样本。这里的长度判断是字符串长度，不是 token 长度，所以只是粗略过滤。真正 token 截断在 `DPOTrainer` 内部由 `max_length` 控制。

### 模型加载和 QLoRA

#### 源码

```python
config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=args.trust_remote_code,
    dtype=torch_dtype,
    cache_dir=args.cache_dir
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    dtype=torch_dtype,
    low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
    device_map=args.device_map,
    trust_remote_code=args.trust_remote_code,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    ) if args.qlora else None,
)

for param in filter(lambda p: p.requires_grad, model.parameters()):
    param.data = param.data.to(torch.float32)
```

#### 逐段解释

这里和 SFT 类似，支持 4bit / 8bit 加载。`args.qlora=True` 时传入 `BitsAndBytesConfig`。

最后把可训练参数转 fp32，避免低精度训练不稳定。

### DPOConfig 和 DPOTrainer

#### 源码

```python
training_args = DPOConfig(
    max_length=full_max_length,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    max_steps=args.max_steps,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    learning_rate=args.learning_rate,
    eval_strategy=args.eval_strategy,
    eval_steps=args.eval_steps,
    output_dir=args.output_dir,
    report_to=args.report_to,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_steps=args.warmup_steps,
    optim=args.optim,
    bf16=args.bf16,
    fp16=args.fp16,
    remove_unused_columns=args.remove_unused_columns,
    run_name=f"dpo_v1",
)

peft_config = None
if args.use_peft:
    target_modules = args.target_modules.split(',') if args.target_modules else None
    if target_modules and 'all' in target_modules:
        target_modules = find_all_linear_names(model, int4=args.load_in_4bit, int8=args.load_in_8bit)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

trainer = DPOTrainer(
    model,
    ref_model=None if args.use_peft else deepcopy(model),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config if args.use_peft else None,
)
```

#### 逐段解释

`DPOConfig` 是 TRL 里的训练参数类，类似 `TrainingArguments`，但专门给 DPO 用。

`max_length=full_max_length` 来自：

```python
full_max_length = max_source_length + max_target_length
```

也就是说 prompt 和 response 的总长度不能超过这个值。

`ref_model=None if args.use_peft else deepcopy(model)` 很重要。使用 PEFT 时，TRL 可以在内部处理 reference；全参训练时，脚本会复制一个冻结参考模型。

`DPOTrainer` 接收：

- `model`：当前要训练的 policy。
- `ref_model`：参考模型。
- `train_dataset`：包含 prompt/chosen/rejected。
- `peft_config`：LoRA 配置。

### 训练和保存

#### 源码

```python
if args.do_train:
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = max_train_samples
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        trainer.model.save_pretrained(args.output_dir)

if args.do_eval:
    metrics = trainer.evaluate()
    metrics["eval_samples"] = max_eval_samples
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
```

#### 逐段解释

DPO 训练后同样保存 adapter。日志里通常会看到 DPO 相关指标，比如 rewards、margin、loss 等，具体字段取决于 TRL 版本。

## 和当前 Qwen3 医疗项目的关系

你现在已经有 SFT 数据，但还没有 preference 数据。要做 DPO，需要新增一个构造数据阶段：

```text
复杂病例 prompt
  -> 大模型生成多个候选答案
  -> 人工或规则判断 chosen/rejected
  -> data/preference_medical/train.jsonl
```

推荐 schema：

```json
{
  "conversations": [
    {"from": "human", "value": "患者主诉胸痛伴出汗，应如何初步判断？"}
  ],
  "chosen": "需要优先排查急性冠脉综合征，建议立即评估生命体征...",
  "rejected": "可能只是劳累，回家休息即可。"
}
```

## 运行命令

```bash
CUDA_VISIBLE_DEVICES=0 python training/dpo_training.py \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --train_file_dir data/preference_medical \
  --validation_file_dir data/preference_medical \
  --template_name qwen3 \
  --do_train \
  --do_eval \
  --use_peft True \
  --qlora True \
  --load_in_4bit True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_source_length 2048 \
  --max_target_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --max_steps 1000 \
  --output_dir outputs/qwen3_4b_medical_dpo \
  --bf16 True \
  --fp16 False \
  --report_to swanlab
```

## 常见坑

- DPO 数据不是 SFT 数据，必须有 `chosen` 和 `rejected`。
- chosen/rejected 必须对应同一个 prompt。
- 不要把 C-Eval 验证集答案直接构造成 DPO 训练标签。
- 当前脚本 `run_name` 写死为 `dpo_v1`，如果你想在 SwanLab 区分实验，可能要改源码或看当前 TRL 参数是否覆盖。
- 字符串长度过滤不是 token 长度过滤，长样本仍可能在 tokenizer 阶段被截断。

## 学习检查清单

- 能解释 DPO 为什么不需要显式 reward model。
- 能说清楚 chosen/rejected/prompt 三列的关系。
- 能看懂 `_build_prompt_from_conversations()` 为什么不能把 chosen 塞进 prompt。
- 能解释 `ref_model` 的作用。
- 能区分 SFT、DPO 和后续 PPO/GRPO。
