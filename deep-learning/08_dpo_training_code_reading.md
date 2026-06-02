# MedicalGPT 训练源码精读 08：dpo_training.py

## 整体作用

DPO 是 Direct Preference Optimization。它不是让模型学习一个标准答案，而是让模型学会：

```text
chosen 的概率 > rejected 的概率
```

数据格式通常是：

```json
{
  "conversations": [{"from": "human", "value": "问题"}],
  "chosen": "更好的回答",
  "rejected": "更差的回答"
}
```

## 参数类：`ScriptArguments`

```python
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default=None)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    train_file_dir: Optional[str] = field(default=None)
    validation_file_dir: Optional[str] = field(default=None)
    template_name: Optional[str] = field(default=None)
    per_device_train_batch_size: Optional[int] = field(default=4)
    max_source_length: Optional[int] = field(default=2048)
    max_target_length: Optional[int] = field(default=512)
    use_peft: bool = field(default=True)
    qlora: bool = field(default=False)
    lora_rank: Optional[int] = field(default=8)
    learning_rate: Optional[float] = field(default=5e-4)
    output_dir: Optional[str] = field(default="outputs-dpo")
```

解释：

- DPO 参数把模型、数据、LoRA、训练参数放在一个 dataclass 里。
- 和 SFT 不同，DPO 需要 preference 数据。
- `max_source_length` 控制 prompt 长度。
- `max_target_length` 控制 chosen/rejected 长度。

## 可训练参数统计

```python
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
```

解释：

LoRA 训练只更新少量参数，这个函数用来确认 PEFT 是否真的生效。

## 自动找 LoRA 线性层

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

解释：

和 SFT 逻辑相同。DPO 也可以做 LoRA / QLoRA。

## DPO 主流程

核心流程可以概括为：

```text
解析参数
  -> 加载 tokenizer
  -> 加载 preference JSONL
  -> conversations 构造 prompt
  -> chosen/rejected 构造成偏好对
  -> 加载模型
  -> 配置 LoRA / QLoRA
  -> DPOTrainer 训练
```

源码里会使用 TRL 的 DPO 训练器，核心思想是：同一个 prompt 下，提高 chosen 的概率，降低 rejected 的相对概率。

## SFT 和 DPO 的区别

SFT 数据：

```json
{"conversations":[{"from":"human","value":"问题"},{"from":"gpt","value":"答案"}]}
```

DPO 数据：

```json
{"conversations":[{"from":"human","value":"问题"}],"chosen":"好回答","rejected":"差回答"}
```

SFT 学：

```text
给定问题，模仿答案
```

DPO 学：

```text
给定问题，更偏好 chosen，不偏好 rejected
```

## 和当前项目的关系

你当前阶段先做 SFT。后面如果要做医疗偏好对齐，可以构造：

- chosen：医学准确、格式规范、安全的回答
- rejected：事实错误、格式差、安全性差的回答

然后用 DPO 继续训练 Qwen3 adapter。

## 常见坑

- DPO 不能直接吃 SFT 数据，必须有 `chosen/rejected`。
- chosen 和 rejected 长度差异太大时，训练可能偏向长度。
- DPO 学的是偏好，不等于事实校验。
- 如果没有高质量偏好数据，DPO 可能不如 SFT 稳。

