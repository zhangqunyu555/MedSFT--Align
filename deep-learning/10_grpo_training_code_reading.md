# MedicalGPT 训练源码精读 10：grpo_training.py

## 整体作用

`MedicalGPT/training/grpo_training.py` 是基于 TRL `GRPOTrainer` 的强化学习训练脚本。它和 `ppo_training.py` 最大区别是：GRPO 可以直接使用 Python 奖励函数，比如格式奖励、答案准确率奖励，而不一定先训练一个 reward model。

当前源码里已经内置两个奖励函数：

```text
accuracy_reward：答案准确率奖励
format_reward：格式合规奖励
```

这和你的项目目标高度相关，尤其是“复杂病例格式回答准确率从 72% 提升到 94%”。

## 方法原理

GRPO 可以理解为 PPO 的一种组相对优化方法。它对同一个 prompt 采样一组回答，然后比较这些回答的 reward 高低，用组内相对优势更新模型。

直观流程：

```text
一个问题
  -> 模型生成多条回答
  -> 每条回答计算 reward
  -> 高于组平均的回答增强
  -> 低于组平均的回答削弱
```

它不需要单独 value model，所以比 PPO 更轻一些。对医学项目来说，GRPO 适合这种可规则评分的目标：

- 是否输出指定格式。
- 是否包含 `<think>` 和 `<answer>`。
- 答案是否能被解析并和标准答案匹配。
- 医学问答里是否包含必要安全提醒。

## 脚本输入输出

当前源码默认支持 GSM8K 风格数据：

```json
{"question": "问题", "answer": "标准答案"}
```

也支持本地目录：

```text
--train_file_dir data/grpo_medical
```

输出：

```text
outputs/qwen3_4b_medical_grpo/
  adapter 或模型权重
  tokenizer files
  trainer_state.json
```

## 主流程图

```text
解析 ModelConfig / ScriptArguments / GRPOConfig
  -> 加载 tokenizer
  -> 加载 question/answer 数据
  -> map 成 chat prompt + answer
  -> 加载 Qwen3 模型
  -> 配置 4bit / LoRA
  -> 注册 accuracy_reward 和 format_reward
  -> GRPOTrainer 训练
  -> 保存模型和 tokenizer
```

## 关键源码精读

### 参数类：ScriptArguments

#### 源码

```python
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with GRPO
    """
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    # Dataset arguments
    dataset_name: Optional[str] = field(
        default="openai/gsm8k",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing training files for local datasets."}
    )
    train_samples: Optional[int] = field(default=-1, metadata={"help": "Number of samples to train on, -1 for all"})
    subset_name: Optional[str] = field(default="main")
    dataset_splits: Optional[str] = field(default="train")
    preprocessing_num_workers: Optional[int] = field(default=10)
    # QLoRA arguments
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
```

#### 逐段解释

GRPO 脚本的数据格式和 SFT/DPO 不一样。它默认从 `openai/gsm8k` 读取 `question/answer`。

如果要用于医疗复杂病例，需要把数据整理成：

```json
{"question": "病例问题", "answer": "标准答案或可验证答案"}
```

`qlora` 表示是否用 4bit base model + LoRA。

### 文本标准化：normalize_text()

#### 源码

```python
def normalize_text(text):
    """Normalize text by removing extra whitespace, converting to lowercase."""
    if text is None:
        return ""
    # Remove extra whitespace and convert to lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text
```

#### 逐段解释

这个函数把文本做轻量标准化：

- `None` 转成空字符串。
- `strip()` 去掉首尾空白。
- `lower()` 转小写。
- `re.sub(r'\s+', ' ', ...)` 把连续空白变成一个空格。

#### 为什么这样写

奖励函数比较答案时，不应该因为多个空格或大小写差异就判错。医学中文里大小写影响不大，但英文缩写如 `CT / ct` 会被统一。

### 答案抽取：extract_answer()

#### 源码

```python
def extract_answer(text):
    """Extract content between <answer> tags."""
    if text is None:
        return ""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
```

#### 逐段解释

GRPO 的系统提示要求模型用：

```text
<think>推理过程</think><answer>最终答案</answer>
```

这个函数从 `<answer>...</answer>` 中抽取最终答案。如果模型没按格式输出，就退回整个文本。

#### 为什么这样写

奖励函数应该主要判断最终答案，不应该把思考过程也拿去和标准答案比较。否则模型推理文字稍微不同就会影响准确率奖励。

### 准确率奖励：accuracy_reward()

#### 源码

```python
def accuracy_reward(completions, answer, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, answer):
        if '####' in sol:
            # for GSM8K
            gold_parsed = parse(sol.split("####", 1)[-1].strip())
            answer_parsed = parse(extract_answer(content))
        else:
            # First try latex parsing
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            logger.warning(f"Error in verification: {e}")
            reward = 0.0
        logger.debug(f"predict_answer: {content}, \nground_truth: {sol}, \n"
                     f"answer_parsed: {answer_parsed}, gold_parsed: {gold_parsed}, reward: {reward}\n\n")
        rewards.append(reward)
    logger.debug(f'accuracy rewards: {rewards}')
    return rewards
```

#### 逐段解释

第一行：

```python
contents = [completion[0]["content"] for completion in completions]
```

TRL 传进来的 `completions` 是一个列表，每个 completion 又是 chat message 结构。这里取出模型生成的文本。

然后遍历模型输出和标准答案：

```python
for content, sol in zip(contents, answer):
```

如果标准答案里有 `####`，按 GSM8K 格式处理，只取 `####` 后面的最终答案。

否则走 latex parsing：

```python
gold_parsed = parse(sol, ...)
answer_parsed = parse(content, ...)
```

最后用：

```python
reward = float(verify(answer_parsed, gold_parsed))
```

判断模型答案和标准答案是否等价。等价给 1.0，不等价给 0.0。

#### 关键变量

- `content`：模型生成内容。
- `sol`：标准答案。
- `gold_parsed`：解析后的标准答案。
- `answer_parsed`：解析后的模型答案。
- `reward`：奖励分数。

#### 和医学任务的关系

这个函数原本偏数学答案验证。医学开放问答不一定能用 latex parse 直接判断。要迁移到复杂病例，需要改成医学奖励，例如：

- 关键词覆盖奖励。
- 语义相似度奖励。
- 诊疗安全规则奖励。
- 大模型裁判奖励。
- C-Eval 选择题答案匹配奖励。

### 格式奖励：format_reward()

#### 源码

```python
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]

    rewards = [1.0 if match else 0.0 for match in matches]
    logger.debug(f'format rewards: {rewards}')
    return rewards
```

#### 逐段解释

`pattern` 要求输出严格符合：

```text
<think>...</think><answer>...</answer>
```

`re.match()` 从字符串开头匹配。源码里的 `$` 要求 `<answer>` 后面就是结尾。

匹配成功给 1.0，失败给 0.0。

#### 为什么适合你的项目

你项目目标里有“复杂病例格式回答准确率 72% -> 94%”。这个函数就是格式奖励的雏形。你可以把格式改成医疗回答模板，例如：

```text
<analysis>病情分析</analysis>
<advice>处理建议</advice>
<safety>就医提醒</safety>
```

然后用正则给格式合规奖励。

### 系统提示：SYSTEM_PROMPT

#### 源码

```python
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
```

#### 逐段解释

这个 prompt 直接告诉模型输出格式：

```text
先思考，再回答。
思考放在 <think>。
答案放在 <answer>。
```

GRPO 的格式奖励和这个系统提示是配套的。提示告诉模型怎么做，奖励函数强化模型真的这么做。

### 数据准备：question/answer 转 chat prompt

#### 源码

```python
if script_args.train_file_dir and os.path.exists(script_args.train_file_dir):
    dataset = load_dataset("json", data_dir=script_args.train_file_dir, split="train")
else:
    dataset = load_dataset(script_args.dataset_name, script_args.subset_name, split=script_args.dataset_splits)

if script_args.train_samples > 0:
    dataset = dataset.shuffle(seed=42).select(range(script_args.train_samples))

with training_args.main_process_first(desc="Dataset preparation"):
    dataset = dataset.map(
        lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': x['answer']
        },
        num_proc=script_args.preprocessing_num_workers,
        desc="Processing dataset" if is_main_process else None,
    )

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]
```

#### 逐段解释

数据来源有两种：

- 本地 `train_file_dir`
- HuggingFace dataset

`map()` 把每条样本转成 TRL GRPO 需要的格式：

```python
{
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": x["question"]}
    ],
    "answer": x["answer"]
}
```

这里的 `prompt` 不是字符串，而是 chat messages 列表。

最后切 10% 做测试集。

#### 医学数据怎么适配

你需要把复杂病例整理成：

```json
{"question": "患者男，65岁，胸痛伴大汗30分钟，应该如何处理？", "answer": "应优先考虑急性冠脉综合征，立即就医并评估心电图和肌钙蛋白..."}
```

### 模型量化加载

#### 源码

```python
torch_dtype = (
    model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
)

if model_args.load_in_4bit and model_args.load_in_8bit:
    raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")

quantization_config = None
if script_args.qlora and (model_args.load_in_4bit or model_args.load_in_8bit):
    if is_deepspeed_zero3_enabled():
        raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=model_args.load_in_4bit,
        load_in_8bit=model_args.load_in_8bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    )

model_kwargs = dict(
    revision=model_args.model_revision,
    trust_remote_code=model_args.trust_remote_code,
    attn_implementation=model_args.attn_implementation,
    dtype=torch_dtype,
    low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
    quantization_config=quantization_config,
)
```

#### 逐段解释

这段和 SFT/QLoRA 类似。GRPO 也支持 4bit 量化加载。

`attn_implementation` 可以控制 attention 实现，例如 flash attention，取决于当前 transformers 和显卡环境。

### 多卡 device_map

#### 源码

```python
num_gpus = torch.cuda.device_count()
if ddp:
    model_kwargs["device_map"] = None
elif num_gpus > 1:
    max_memory = {}
    for i in range(num_gpus):
        gpu_props = torch.cuda.get_device_properties(i)
        total_mem = gpu_props.total_memory
        usable_mem = int(total_mem * 0.8)
        max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"
    model_kwargs["max_memory"] = max_memory
    model_kwargs["device_map"] = "auto"
else:
    model_kwargs["device_map"] = "auto"

model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    **model_kwargs,
)
```

#### 逐段解释

单卡或单进程多卡用 `device_map=auto`。DDP 时不用 device_map，让分布式训练框架接管。

### LoRA 配置

#### 源码

```python
if model_args.use_peft:
    if training_args.gradient_checkpointing:
        logger.warning("Gradient checkpointing is enabled. It may cause issues with LoRA, setting it to False.")
        training_args.gradient_checkpointing = False
    target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
    if target_modules == 'all' or (target_modules and 'all' in target_modules):
        target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )
    model = get_peft_model(model, peft_config)
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)
    model.print_trainable_parameters()
else:
    logger.info("Fine-tuning method: Full parameters training")
```

#### 逐段解释

这里用的是 TRL `ModelConfig` 里的 LoRA 参数，字段名和 SFT 脚本略有不同：

- SFT 用 `lora_rank`
- GRPO 这里用 `lora_r`

源码还会在 LoRA 时关闭 gradient checkpointing，因为当前实现认为两者可能冲突。

### GRPOTrainer 初始化

#### 源码

```python
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        accuracy_reward,
        format_reward
    ],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
)
```

#### 逐段解释

这是 GRPO 的核心。

`reward_funcs` 传入两个 Python 函数：

```python
[
    accuracy_reward,
    format_reward
]
```

训练时，每个 completion 会得到两个 reward。TRL 会把它们组合起来作为优化信号。

#### 为什么 GRPO 对你有用

你想优化“格式 + 准确率”的多维奖励，GRPO 的接口天然适合：

```python
reward_funcs=[
    medical_accuracy_reward,
    medical_format_reward,
    safety_reward,
]
```

相比 PPO/RLOO，它不一定需要先训练 reward model。

### 训练和保存

#### 源码

```python
last_checkpoint = get_checkpoint(training_args)
if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    if is_main_process:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

if is_main_process:
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

trainer.model.config.use_cache = True
if is_main_process:
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

training_args.distributed_state.wait_for_everyone()

if is_main_process:
    tokenizer.save_pretrained(training_args.output_dir)
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["r1", "grpo"],
    }
    trainer.create_model_card(**kwargs)
    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(training_args.output_dir)
```

#### 逐段解释

`get_checkpoint()` 会检查输出目录里有没有旧 checkpoint，有的话自动续训。

训练后保存：

- train metrics
- trainer state
- model / adapter
- tokenizer
- model card
- config

`wait_for_everyone()` 用在分布式训练，确保所有进程同步。

## 和当前 Qwen3 医疗项目的关系

GRPO 是你后续强化对齐里最值得重点看的脚本，因为你项目目标明确包含：

```text
回答格式 + 医学准确率 的多维奖励函数
```

当前源码的格式奖励可以直接借鉴，但准确率奖励需要改成医学版。建议后续设计：

```text
medical_format_reward：
  检查是否包含病情分析、处理建议、风险提示

medical_accuracy_reward：
  用 embedding / LLM judge / 标准答案关键词判断医学正确性

safety_reward：
  检查是否避免绝对化诊断，是否建议必要就医
```

## 运行命令

示例：

```bash
CUDA_VISIBLE_DEVICES=0 python training/grpo_training.py \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --train_file_dir data/grpo_medical \
  --output_dir outputs/qwen3_4b_medical_grpo \
  --qlora True \
  --load_in_4bit True \
  --use_peft True \
  --lora_target_modules all \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-6 \
  --num_train_epochs 1 \
  --bf16 True \
  --report_to swanlab
```

参数名来自 TRL `ModelConfig / GRPOConfig`，和 SFT 脚本不完全一致，正式跑前建议先：

```bash
python training/grpo_training.py --help
```

## 常见坑

- GRPO 数据不是 ShareGPT，而是 `question/answer`。
- 当前 `accuracy_reward` 偏数学题，不能直接用于开放医学问答。
- `format_reward` 的正则很严格，输出后多一个换行都可能失败。
- LoRA 参数名和 SFT 不完全一样。
- GRPO 会生成多条回答，显存压力大于普通 SFT。
- 医疗奖励函数设计不好，会强化错误医学行为。

## 学习检查清单

- 能解释 GRPO 和 PPO/RLOO 的区别。
- 能看懂 `reward_funcs=[accuracy_reward, format_reward]` 的意义。
- 能说明 `<think>/<answer>` 格式奖励怎么计算。
- 能指出当前准确率奖励为什么不适合直接评医学开放问答。
- 能设计一个医学复杂病例的 reward 函数方向。
