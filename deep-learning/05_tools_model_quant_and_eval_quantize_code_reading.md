# MedicalGPT 工具源码精读 05：model_quant.py 与 eval_quantize.py

## 整体作用

这篇合并学习两个量化相关工具：

- `model_quant.py`：加载未量化模型，做 4bit 量化，比较推理速度和显存。
- `eval_quantize.py`：加载量化模型，在 JSONL 数据上计算 PPL。

注意：量化推理和 QLoRA 训练不是一回事。

```text
量化推理：为了部署/推理省显存
QLoRA：训练时 4bit 加载底座模型，只训练 LoRA adapter
```

## model_quant.py 源码精读

### 参数解析

```python
def parse_args():
    parser = argparse.ArgumentParser(description="量化模型推理对比")
    parser.add_argument("--unquantized_model_path", type=str, required=True, help="未量化模型路径")
    parser.add_argument("--quantized_model_output_path", type=str, required=True, help="量化模型保存路径")
    parser.add_argument("--input_text", type=str, default='介绍北京', help="输入的文本内容")
    return parser.parse_args()
```

解释：

- `--unquantized_model_path`：原始模型路径。
- `--quantized_model_output_path`：量化后保存路径。
- `--input_text`：测试推理用的问题。

### 显存占用函数

```python
def get_model_memory_usage(device):
    return torch.cuda.memory_allocated(device) / (1024 ** 3)
```

解释：

- `torch.cuda.memory_allocated(device)` 返回当前 GPU 已分配显存。
- 除以 `1024 ** 3` 转成 GB。

### 推理函数

```python
def perform_inference(model, tokenizer, devic, question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)
    attention_mask = inputs["attention_mask"]

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    elapsed_time = end_time - start_time

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text, elapsed_time
```

逐段解释：

- tokenizer 把文本转成 tensor。
- `.to(device)` 放到 GPU。
- `torch.no_grad()` 表示推理不计算梯度。
- `model.generate()` 生成回答。
- `temperature`、`top_p` 控制采样。
- `repetition_penalty` 降低重复。
- 最后 decode 成文本，并返回耗时。

### 4bit 量化配置

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.uint8,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
```

解释：

- `load_in_4bit=True`：启用 4bit。
- `bnb_4bit_quant_type="nf4"`：使用 NF4，这是 QLoRA 常用量化类型。
- `bnb_4bit_use_double_quant=True`：双重量化，进一步节省显存。
- `bnb_4bit_compute_dtype=torch.float16`：计算时用 fp16。

## eval_quantize.py 源码精读

### 设备选择

```python
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

解释：

优先 CUDA，其次 Mac MPS，最后 CPU。

### 读取 JSONL

```python
def load_jsonl_data(file_path):
    conversation_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            conversations = data.get("conversations", [])
            if len(conversations) >= 2:
                user_msg = conversations[0].get("value", "")
                assistant_msg = conversations[1].get("value", "")
                conversation_pairs.append((user_msg, assistant_msg))
    return conversation_pairs
```

解释：

它把 ShareGPT 格式转成 `(用户问题, 助手答案)` 对，后面用于计算 PPL。

### PPL 评估逻辑

```python
def evaluate_perplexity(model, tokenizer, conversation_pairs):
    model = model.eval()
    total_loss = 0
    total_tokens = 0

    for user_msg, assistant_msg in conversation_pairs:
        text = user_msg + assistant_msg
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        num_tokens = inputs["input_ids"].numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl
```

解释：

- 把 prompt 和 answer 拼成文本。
- 用 causal LM 的 `labels=input_ids` 计算自回归 loss。
- loss 乘 token 数累加。
- 平均 loss 后取 `exp` 得到 PPL。

## 和当前项目的关系

当前主线先做 QLoRA SFT，不急着量化模型。量化工具适合后续：

- 合并 LoRA 后做量化部署。
- 比较量化前后 PPL。
- 看 4bit 模型是否明显退化。

## 常见坑

- bitsandbytes 主要适配 CUDA，Mac MPS 不适合跑 4bit。
- PPL 计算方式会受 prompt 拼接方式影响，最好和训练模板一致。
- 量化保存的模型不应提交 Git。

