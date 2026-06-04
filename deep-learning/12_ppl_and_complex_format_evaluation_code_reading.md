# PPL 与复杂病例格式回答准确率评测源码精读

这一篇学习文档专门讲项目最后两个指标怎么测：

- `1K 医疗长文本 PPL`
- `复杂病例格式回答准确率`

这两个指标和 C-Eval 不一样。C-Eval 测的是选择题准确率，偏知识和考试能力；PPL 测的是模型对医疗长文本参考答案的拟合程度；复杂病例格式回答准确率测的是模型有没有按照我们 PPO 阶段要求的四段式医学回答格式输出。

本阶段涉及四个脚本：

```text
scripts/build_ppl_eval_set.py
scripts/evaluate_medical_ppl.py
scripts/evaluate_complex_case_format.py
scripts/add_format_prompt_to_ppo_dataset.py
```

整体流程是：

```text
cleaned_alpaca.jsonl
  -> build_ppl_eval_set.py
  -> medical_longtext_ppl_1k.jsonl
  -> evaluate_medical_ppl.py
  -> eval_results/ppl/*.json

medical_complex_cases_5k.jsonl
  -> add_format_prompt_to_ppo_dataset.py
  -> 带四段式格式提示的 medical_complex_cases_5k.jsonl
  -> evaluate_complex_case_format.py
  -> responses.jsonl
  -> report.json
  -> prompted format accuracy
```

## 一、整体目标

你现在已经完成了：

- 50 万候选数据准备
- 清洗后 381621 条医疗 SFT 数据
- C-Eval 医学目标集构建
- 相似度筛选 Top 100000
- Qwen3 LoRA / QLoRA SFT
- PPO 多维奖励强化对齐
- C-Eval 医学评测

最后缺的是：

```text
1K 医疗长文本 PPL：
  用固定的 1000 条长参考答案，比较 base / SFT / PPO 三个模型的困惑度。

复杂病例格式回答准确率：
  用 5K 复杂病例 prompt，让模型生成回答，检查是否包含：
  病情分析 / 处理建议 / 风险提示 / 就医建议
```

这两个指标可以支撑简历里的这类表达：

```text
将 1K 条专业医疗长文本 PPL 由 15.194 降至 9.823。

复杂病例格式回答准确率由 72% 提升至 94%。
```

注意，具体数值要以真实跑出来的报告为准。脚本只是提供可复现的测量方法。

## 二、指标原理

### 1. PPL 是什么

PPL 全称是 perplexity，中文一般叫困惑度。

语言模型训练时本质是在预测下一个 token。假设参考答案 token 是：

```text
y1, y2, y3, ..., yn
```

模型会给每个 token 一个概率：

```text
P(y1), P(y2 | y1), P(y3 | y1,y2), ...
```

如果模型认为参考答案很自然、很容易生成，那么这些 token 的概率就高，loss 就低，PPL 就低。

PPL 的核心公式可以直观理解为：

```text
eval_loss = 所有答案 token 的平均负对数似然
ppl = exp(eval_loss)
```

所以：

- PPL 越低，说明模型越熟悉这类文本。
- SFT 后 PPL 下降，说明模型更适应医疗回答风格。
- PPO 后 PPL 不一定继续下降，因为 PPO 优化的是奖励函数，不是纯语言建模 loss。

### 2. 为什么要 answer-only PPL

我们的评测样本是 Alpaca 格式：

```json
{
  "instruction": "妊娠后期水肿的鉴别诊断",
  "input": "",
  "output": "根据病史、水肿的表现及上述的各项检查..."
}
```

如果直接把 prompt 和 answer 全部算 PPL，会出现一个问题：

```text
模型也会被要求预测用户问题。
```

但 SFT 训练真正关心的是：

```text
给定用户问题后，assistant 的回答写得像不像参考答案。
```

所以脚本中使用：

```python
labels = [-100] * len(prompt_ids) + answer_ids
```

`-100` 是 HuggingFace 交叉熵默认忽略标签。也就是说：

- prompt token 参与上下文输入
- prompt token 不参与 loss
- answer token 参与 loss
- 最后 PPL 只反映答案部分

这就是 answer-only PPL。

### 3. 复杂病例格式回答准确率是什么

PPO 阶段的目标不是只让模型答得像参考答案，还要让输出符合复杂病例格式。

我们要求回答包含四段：

```text
病情分析
处理建议
风险提示
就医建议
```

如果四段都出现：

```text
format_score = 1.0
format_pass = True
```

如果只出现两段：

```text
format_score = 0.5
format_pass = False
```

最终格式准确率：

```text
format_accuracy = format_pass_count / num_samples
```

这就是“复杂病例格式回答准确率”。

### 4. 辅助指标

格式准确率是主指标，但脚本还会输出三个辅助指标：

```text
avg_format_score
safety_coverage
keyword_coverage
```

含义是：

```text
avg_format_score：
  平均格式得分。即使没有完全通过，也能看出平均覆盖了几段。

safety_coverage：
  高风险病例是否有及时就医、急诊、医生评估等安全提示。

keyword_coverage：
  生成回答中覆盖了多少参考答案关键词。
```

`keyword_coverage` 不是严格医学准确率，只是辅助观察，因为开放医学问答不能简单靠关键词完全判断正确。

### 5. 格式 Prompt 口径说明

这次你看到 `responses.jsonl` 后发现了一个非常关键的问题：

```text
模型回答了医学内容，但没有出现四个固定标题：
病情分析 / 处理建议 / 风险提示 / 就医建议
```

因此评测脚本会全部判成：

```json
{
  "format_score": 0.0,
  "missing_sections": ["病情分析", "处理建议", "风险提示", "就医建议"]
}
```

这不是模型完全不会回答医学问题，而是：

```text
评测规则要求四段式标题，但输入 prompt 没明确要求模型按四段式标题回答。
```

所以后来我们把复杂病例数据的 `prompt` 改成：

```text
请严格按照以下四个小标题回答：
1. 病情分析
2. 处理建议
3. 风险提示
4. 就医建议

病例问题：{原始病例问题}
```

这样做是合理的，但指标口径必须讲清楚。

更严谨的说法是：

```text
带格式提示的复杂病例格式回答准确率
```

或者：

```text
复杂病例格式指令遵循准确率
```

不能把它说成：

```text
模型在无格式提示下自发输出四段式回答。
```

如果以后要做更严谨的实验，可以同时报告两列：

```text
unprompted format accuracy：
  不加格式提示，测模型是否自发输出四段式。

prompted format accuracy：
  加格式提示，测模型是否能按指令稳定输出四段式。
```

当前项目为了复现简历中“复杂病例格式回答准确率提升”的指标，正式使用的是带格式提示版本。

## 三、真实数据与产物

当前本地已经生成：

```text
data/eval/medical_longtext_ppl_1k.jsonl
```

行数：

```text
1000
```

来源：

```text
data/cleaned/shibing624_medical_500k/cleaned_alpaca.jsonl
```

抽取策略：

```text
按 output_char_length 从长到短排序，取前 1000 条。
```

当前真实统计：

```text
selected_min_output_chars = 2131
selected_max_output_chars = 8199
selected_avg_output_chars = 3384.33
```

复杂病例数据：

```text
data/rl/medical_complex_cases_5k.jsonl
```

行数：

```text
5000
```

每条数据核心字段：

```json
{
  "prompt": "复杂病例问题",
  "reference_answer": "参考答案",
  "answer_keywords": ["关键词1", "关键词2"],
  "risk_level": "high",
  "required_sections": ["病情分析", "处理建议", "风险提示", "就医建议"]
}
```

当前这个文件已经被 `add_format_prompt_to_ppo_dataset.py` 覆盖为带格式提示版本。为了可回退，原始无格式提示版本备份在：

```text
data/rl/medical_complex_cases_5k_no_format_prompt_backup.jsonl
```

覆盖后的样本会多三个字段：

```json
{
  "format_prompt_enabled": true,
  "original_prompt": "原始病例问题",
  "format_instruction": "请严格按照以下四个小标题回答：..."
}
```

其中：

- `prompt`：已经变成 `format_instruction + original_prompt`
- `original_prompt`：保留原始病例问题
- `format_prompt_enabled`：标记这条数据已经加过格式提示，防止重复添加

## 四、`build_ppl_eval_set.py` 源码精读

这个脚本的作用是从清洗后的 381621 条 Alpaca 数据中，构建固定的 1000 条长文本 PPL 评测集。

输入：

```text
data/cleaned/shibing624_medical_500k/cleaned_alpaca.jsonl
```

输出：

```text
data/eval/medical_longtext_ppl_1k.jsonl
```

### 函数：`parse_args()`

#### 这段代码完整实现

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed medical long-text PPL eval set.")
    parser.add_argument("--input", required=True, help="Input Alpaca JSONL file.")
    parser.add_argument("--output", required=True, help="Output PPL eval JSONL file.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of samples to keep.")
    parser.add_argument(
        "--min-output-chars",
        type=int,
        default=200,
        help="Drop answers shorter than this before ranking by length.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50000,
        help="Print progress every N rows.",
    )
    return parser.parse_args()
```

#### 这段代码在做什么

`parse_args()` 定义命令行参数，让脚本可以灵活指定输入、输出、样本数和进度日志间隔。

#### 逐段解释

第一段：

```python
parser = argparse.ArgumentParser(...)
```

创建命令行解析器。这个对象负责读取你在终端里传入的参数。

第二段：

```python
parser.add_argument("--input", required=True, ...)
parser.add_argument("--output", required=True, ...)
```

这两个参数必须传：

- `--input`：原始 Alpaca JSONL
- `--output`：输出 PPL 评测集

第三段：

```python
parser.add_argument("--sample-size", type=int, default=1000, ...)
```

默认取 1000 条，正好对应简历里的 `1K 条专业医疗长文本 PPL`。

第四段：

```python
parser.add_argument("--min-output-chars", type=int, default=200, ...)
```

先过滤掉答案太短的样本。虽然最终我们按长度取 Top 1000，实际最短已经达到 2131 字符，但这个参数可以防止数据集变小时混入短回答。

第五段：

```python
parser.add_argument("--log-every", type=int, default=50000, ...)
```

每读 50000 行打印一次进度。因为 381621 行不算小，长时间脚本必须有进度日志。

#### 为什么这样写

这个脚本要在本机和服务器都能跑，不依赖外部配置文件，所以直接用命令行参数最稳。默认值也贴合当前项目目标：1000 条、长文本、可复现。

### 函数：`iter_alpaca_rows()`

#### 这段代码完整实现

```python
def iter_alpaca_rows(path: Path, log_every: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_at = time.time()
    rows: list[dict[str, Any]] = []
    total = 0
    bad_json = 0
    missing_fields = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            instruction = clean_text(row.get("instruction"))
            input_text = clean_text(row.get("input"))
            output = clean_text(row.get("output"))
            if not (instruction or input_text) or not output:
                missing_fields += 1
                continue

            rows.append(
                {
                    "id": row.get("id") or f"ppl-candidate-{line_no}",
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                    "source": row.get("source", "cleaned_alpaca"),
                    "source_row_id": row.get("row_id", row.get("source_row_id", line_no)),
                    "output_char_length": len(output),
                    "input_char_length": len(instruction) + len(input_text),
                }
            )

            if log_every > 0 and total % log_every == 0:
                elapsed = time.time() - started_at
                speed = total / elapsed if elapsed > 0 else 0.0
                print(
                    f"[progress] read={total} valid={len(rows)} bad_json={bad_json} "
                    f"missing={missing_fields} speed={speed:.1f} rows/s",
                    flush=True,
                )

    report = {
        "input": str(path),
        "total_read": total,
        "valid_candidates": len(rows),
        "bad_json": bad_json,
        "missing_fields": missing_fields,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    return rows, report
```

#### 这段代码在做什么

这个函数逐行读取 Alpaca JSONL，把合法样本整理成统一结构，并记录读取报告。

#### 逐段解释

初始化：

```python
started_at = time.time()
rows: list[dict[str, Any]] = []
total = 0
bad_json = 0
missing_fields = 0
```

这里准备了四类统计：

- `total`：总读取行数
- `bad_json`：JSON 解析失败数量
- `missing_fields`：缺少问题或答案的数量
- `rows`：最终可参与 PPL 候选的样本

读取 JSONL：

```python
with path.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
```

JSONL 是一行一个 JSON。这里使用逐行读取，不一次性把文件全部读成字符串，适合大文件。

清理空行：

```python
line = line.strip()
if not line:
    continue
```

如果某行是空的，不参与统计候选。

解析 JSON：

```python
try:
    row = json.loads(line)
except json.JSONDecodeError:
    bad_json += 1
    continue
```

遇到坏 JSON 不让整个脚本崩掉，只统计并跳过。

字段清洗：

```python
instruction = clean_text(row.get("instruction"))
input_text = clean_text(row.get("input"))
output = clean_text(row.get("output"))
```

Alpaca 格式核心字段是：

```text
instruction
input
output
```

这里允许 `input` 为空，但不允许问题和答案都空。

字段校验：

```python
if not (instruction or input_text) or not output:
    missing_fields += 1
    continue
```

如果没有用户问题，或者没有参考答案，就不能用于 PPL。

构造候选样本：

```python
rows.append(
    {
        "id": row.get("id") or f"ppl-candidate-{line_no}",
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "source": row.get("source", "cleaned_alpaca"),
        "source_row_id": row.get("row_id", row.get("source_row_id", line_no)),
        "output_char_length": len(output),
        "input_char_length": len(instruction) + len(input_text),
    }
)
```

这里除了保留训练字段，还加入两个长度字段：

- `output_char_length`：后面按它选长文本
- `input_char_length`：方便分析 prompt 长度

进度日志：

```python
if log_every > 0 and total % log_every == 0:
    ...
    print(...)
```

每隔一定行数输出：

```text
read=读取行数
valid=有效候选数
bad_json=坏 JSON 数
missing=缺字段数
speed=处理速度
```

最终报告：

```python
report = {
    "input": str(path),
    "total_read": total,
    "valid_candidates": len(rows),
    "bad_json": bad_json,
    "missing_fields": missing_fields,
    "elapsed_seconds": round(time.time() - started_at, 3),
}
```

这份报告能证明数据集构建过程可复现。

#### 为什么这样写

PPL 评测集必须固定，否则每次测出来不能比较。这个函数先把所有合法候选读取出来，并记录长度，后面才能做确定性排序选择。

### 函数：`build_eval_set()`

#### 这段代码完整实现

```python
def build_eval_set(input_path: Path, output_path: Path, sample_size: int, min_output_chars: int, log_every: int) -> dict[str, Any]:
    rows, report = iter_alpaca_rows(input_path, log_every)
    rows = [row for row in rows if row["output_char_length"] >= min_output_chars]
    rows.sort(key=lambda row: (-row["output_char_length"], str(row["id"])))
    selected = rows[:sample_size]

    if len(selected) != sample_size:
        raise SystemExit(
            f"Not enough long-text samples: need={sample_size}, got={len(selected)}, "
            f"min_output_chars={min_output_chars}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for new_id, row in enumerate(selected):
            row = dict(row)
            row["id"] = f"medical-longtext-ppl-{new_id}"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = [row["output_char_length"] for row in selected]
    report.update(
        {
            "output": str(output_path),
            "sample_size": sample_size,
            "min_output_chars": min_output_chars,
            "selected_count": len(selected),
            "selected_min_output_chars": min(lengths),
            "selected_max_output_chars": max(lengths),
            "selected_avg_output_chars": round(sum(lengths) / len(lengths), 2),
            "selection_strategy": "sort_by_output_char_length_desc_then_id",
        }
    )
    return report
```

#### 这段代码在做什么

这个函数真正完成“选 1000 条长文本”的逻辑。

#### 逐段解释

读取候选：

```python
rows, report = iter_alpaca_rows(input_path, log_every)
```

先拿到全部合法 Alpaca 样本。

过滤短答案：

```python
rows = [row for row in rows if row["output_char_length"] >= min_output_chars]
```

答案太短的样本不适合做“长文本 PPL”。

排序：

```python
rows.sort(key=lambda row: (-row["output_char_length"], str(row["id"])))
```

排序规则是：

1. `output_char_length` 越长越靠前
2. 长度相同按 `id` 排序

这里用 `id` 做第二排序，是为了稳定复现。否则长度一样时，不同 Python 环境里虽然通常稳定，但显式指定更稳。

选前 1000 条：

```python
selected = rows[:sample_size]
```

这一步直接对应 1K PPL 评测集。

数量检查：

```python
if len(selected) != sample_size:
    raise SystemExit(...)
```

如果数据不够 1000 条，脚本直接失败，不生成一个不完整评测集。

写 JSONL：

```python
with output_path.open("w", encoding="utf-8") as f:
    for new_id, row in enumerate(selected):
        row = dict(row)
        row["id"] = f"medical-longtext-ppl-{new_id}"
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

每条样本重新赋予稳定 ID：

```text
medical-longtext-ppl-0
medical-longtext-ppl-1
...
```

这样后面 base / SFT / PPO 三组模型都用完全同一批样本。

更新报告：

```python
report.update(
    {
        "selected_min_output_chars": min(lengths),
        "selected_max_output_chars": max(lengths),
        "selected_avg_output_chars": round(sum(lengths) / len(lengths), 2),
        "selection_strategy": "sort_by_output_char_length_desc_then_id",
    }
)
```

报告里会记录最短、最长、平均长度，以及选择策略。

#### 为什么这样写

PPL 指标非常容易因为评测数据不同而不可比较。这个函数把评测集固定下来，之后所有模型都跑同一份 `medical_longtext_ppl_1k.jsonl`，结果才有对照意义。

## 五、`evaluate_medical_ppl.py` 源码精读

这个脚本用于计算 answer-only PPL。

它支持三种模型形态：

```text
HuggingFace Hub 模型
本地 merged 完整模型
base model + LoRA adapter
```

### 函数：`load_model_and_tokenizer()`

#### 这段代码完整实现

```python
def load_model_and_tokenizer(args: argparse.Namespace):
    torch_dtype = dtype_from_name(args.torch_dtype)
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype if torch_dtype != "auto" else torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto",
    )

    if args.peft_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.peft_path, is_trainable=False)

    model.eval()
    return model, tokenizer
```

#### 这段代码在做什么

这个函数加载 tokenizer 和模型，并根据参数决定是否 4bit 加载、是否加载 LoRA adapter。

#### 逐段解释

dtype 解析：

```python
torch_dtype = dtype_from_name(args.torch_dtype)
```

命令行传的是字符串，比如：

```text
bfloat16
float16
float32
auto
```

模型加载时需要的是 `torch.bfloat16` 这样的类型，所以要转换。

4bit 配置：

```python
if args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(...)
```

如果传：

```bash
--load_in_4bit True
```

就启用 bitsandbytes 4bit 加载。核心参数：

```python
load_in_4bit=True
bnb_4bit_use_double_quant=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16
```

这和 QLoRA 常用设置一致，可以节省显存。

加载 tokenizer：

```python
tokenizer = AutoTokenizer.from_pretrained(...)
```

`trust_remote_code=True` 对 Qwen 系列经常需要打开。

pad token 处理：

```python
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
```

如果 tokenizer 没有 pad token，就用 eos token 兜底。

`padding_side = "left"` 对 decoder-only 模型更稳，尤其是批量生成和长文本评测时。

加载模型：

```python
model = AutoModelForCausalLM.from_pretrained(...)
```

这是因果语言模型加载入口。Qwen3 属于 causal LM，所以用这个类。

关键参数：

```python
device_map="auto"
```

让 transformers 自动把模型放到 GPU。多卡时也可以自动切分。

加载 PEFT adapter：

```python
if args.peft_path:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.peft_path, is_trainable=False)
```

如果你没有合并 LoRA，而是保留 adapter 目录，可以这样加载：

```bash
--model_name_or_path Qwen/Qwen3-4B-Instruct-2507
--peft_path outputs/your_adapter
```

`is_trainable=False` 表示只评测，不训练。

评估模式：

```python
model.eval()
```

关闭 dropout 等训练行为，保证评测稳定。

#### 为什么这样写

你的模型可能有三种状态：

```text
1. 原始 HF 模型
2. SFT 后 merged 模型
3. PPO 后输出模型或 adapter
```

这个函数尽量把三种加载方式统一，避免每个模型写一套评测脚本。

### 函数：`render_prompt()`

#### 这段代码完整实现

```python
def render_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
```

#### 这段代码在做什么

这个函数把普通用户问题转换成 Qwen3 chat prompt。

#### 逐段解释

构造 messages：

```python
messages = [{"role": "user", "content": user_prompt}]
```

HuggingFace chat template 通常接受 OpenAI 风格的 messages：

```json
[
  {"role": "user", "content": "问题"}
]
```

使用 tokenizer 的 chat template：

```python
if hasattr(tokenizer, "apply_chat_template"):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

如果 tokenizer 支持 `apply_chat_template()`，就用模型自带模板。

`add_generation_prompt=True` 很关键。它会在 prompt 末尾加 assistant 开始标记，告诉模型：

```text
下面该 assistant 回答了。
```

兜底模板：

```python
return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
```

如果 tokenizer 没有 chat template，就手写 ChatML 格式。

#### 为什么这样写

训练时用 Qwen3 对话模板，评测时也必须用同一类模板。否则 PPL 会因为 prompt 格式不同产生偏差。

### 函数：`build_answer_only_features()`

#### 这段代码完整实现

```python
def build_answer_only_features(tokenizer: Any, row: dict[str, Any], max_length: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompt_text = render_prompt(tokenizer, build_user_prompt(row))
    answer_text = clean_text(row.get("output"))
    if tokenizer.eos_token:
        answer_text += tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids

    if len(answer_ids) > max_length:
        answer_ids = answer_ids[:max_length]
        prompt_ids = []
    elif len(prompt_ids) + len(answer_ids) > max_length:
        keep_prompt = max_length - len(answer_ids)
        prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if not answer_ids:
        raise ValueError("Empty answer tokens after tokenization.")

    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
        len(answer_ids),
    )
```

#### 这段代码在做什么

这是 PPL 脚本最核心的函数。它把一条 Alpaca 样本转换成模型输入，并设置 labels，使 loss 只计算答案部分。

#### 逐段解释

构造 prompt：

```python
prompt_text = render_prompt(tokenizer, build_user_prompt(row))
```

`build_user_prompt(row)` 会把：

```text
instruction + input
```

拼成用户输入。然后 `render_prompt()` 把它转成 Qwen3 对话模板。

取参考答案：

```python
answer_text = clean_text(row.get("output"))
if tokenizer.eos_token:
    answer_text += tokenizer.eos_token
```

`output` 是参考答案。后面追加 eos token，是为了让模型也学习在答案结束时停止。

分别 tokenize：

```python
prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
```

这里没有把 prompt 和 answer 直接拼字符串再 tokenize，而是分开 tokenize。

原因是后面要知道：

```text
prompt 有多少 token
answer 有多少 token
```

只有知道边界，才能 mask prompt loss。

长度截断：

```python
if len(answer_ids) > max_length:
    answer_ids = answer_ids[:max_length]
    prompt_ids = []
elif len(prompt_ids) + len(answer_ids) > max_length:
    keep_prompt = max_length - len(answer_ids)
    prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
```

模型最大上下文有限，比如 2048。如果 prompt + answer 超过最大长度，就需要截断。

这里优先保留答案，因为 PPL 的目标是算答案 loss。

如果答案本身就超过 `max_length`：

```python
answer_ids = answer_ids[:max_length]
prompt_ids = []
```

如果 prompt + answer 超长，但答案没超：

```python
prompt_ids = prompt_ids[-keep_prompt:]
```

保留 prompt 的末尾。对 chat prompt 来说，末尾通常包含 assistant 起始标记，更重要。

构造输入：

```python
input_ids = prompt_ids + answer_ids
```

模型实际看到的是：

```text
prompt token + answer token
```

构造 labels：

```python
labels = [-100] * len(prompt_ids) + answer_ids
```

这是全脚本最关键的一句。

在 HuggingFace 的 causal LM loss 里：

```text
label = -100 的位置会被忽略
```

所以这句表示：

```text
prompt token 不算 loss
answer token 计算 loss
```

返回张量：

```python
return (
    torch.tensor([input_ids], dtype=torch.long),
    torch.tensor([labels], dtype=torch.long),
    len(answer_ids),
)
```

返回三个东西：

- `input_ids`：喂给模型的 token
- `labels`：计算 loss 的标签
- `len(answer_ids)`：答案 token 数，用于后面加权平均 loss

#### 为什么这里要这样写

如果不 mask prompt，PPL 就会混入“预测用户问题”的难度。SFT 模型不是为了复述用户问题训练的，所以这样测不公平。

answer-only PPL 更接近：

```text
给定医疗问题，模型生成参考医学回答的困惑度。
```

### 函数：`evaluate_ppl()`

#### 这段代码完整实现

```python
def evaluate_ppl(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.time()
    rows = load_rows(Path(args.data_path), args.limit)
    if not rows:
        raise SystemExit(f"No valid rows found in {args.data_path}")

    model, tokenizer = load_model_and_tokenizer(args)
    total_nll = 0.0
    total_answer_tokens = 0
    skipped = 0
    device = next(model.parameters()).device

    iterator = tqdm(rows, desc="PPL", dynamic_ncols=True)
    for index, row in enumerate(iterator, 1):
        try:
            input_ids, labels, answer_tokens = build_answer_only_features(tokenizer, row, args.max_length)
        except Exception:
            skipped += 1
            continue

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        total_nll += float(loss.detach().cpu()) * answer_tokens
        total_answer_tokens += answer_tokens

        if args.log_every > 0 and index % args.log_every == 0 and total_answer_tokens > 0:
            running_loss = total_nll / total_answer_tokens
            iterator.set_postfix(loss=f"{running_loss:.4f}", ppl=f"{math.exp(running_loss):.3f}")

    if total_answer_tokens == 0:
        raise SystemExit("No answer tokens were evaluated.")

    eval_loss = total_nll / total_answer_tokens
    perplexity = math.exp(eval_loss) if eval_loss < 100 else float("inf")
    return {
        "model_name_or_path": args.model_name_or_path,
        "peft_path": args.peft_path,
        "data_path": args.data_path,
        "num_samples": len(rows) - skipped,
        "skipped_samples": skipped,
        "num_answer_tokens": total_answer_tokens,
        "avg_answer_tokens": round(total_answer_tokens / max(len(rows) - skipped, 1), 2),
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "max_length": args.max_length,
        "load_in_4bit": args.load_in_4bit,
        "torch_dtype": args.torch_dtype,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
```

#### 这段代码在做什么

这个函数遍历 1K 评测集，累加每条样本的答案 loss，最后计算整体 PPL。

#### 逐段解释

加载数据：

```python
rows = load_rows(Path(args.data_path), args.limit)
```

`--limit` 可以做 smoke test，比如先跑 20 条。

加载模型：

```python
model, tokenizer = load_model_and_tokenizer(args)
```

这里会加载 base / SFT / PPO 模型。

初始化统计：

```python
total_nll = 0.0
total_answer_tokens = 0
skipped = 0
```

这里要注意，不能简单平均每条样本的 loss。

因为每条答案长度不同，应该按 token 数加权：

```text
总负对数似然 / 总答案 token 数
```

循环评测：

```python
iterator = tqdm(rows, desc="PPL", dynamic_ncols=True)
for index, row in enumerate(iterator, 1):
```

`tqdm` 会显示进度条。长时间评测必须有这个，不然服务器上不知道卡没卡住。

构造输入：

```python
input_ids, labels, answer_tokens = build_answer_only_features(...)
```

这一步完成 prompt mask 和 answer label。

移动到 GPU：

```python
input_ids = input_ids.to(device)
labels = labels.to(device)
```

模型在哪张卡，输入就放到同一个 device。

前向计算：

```python
with torch.no_grad():
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
```

`torch.no_grad()` 表示评测不计算梯度，节省显存。

HuggingFace causal LM 如果传入 `labels`，会自动返回交叉熵 loss。

累加 NLL：

```python
total_nll += float(loss.detach().cpu()) * answer_tokens
total_answer_tokens += answer_tokens
```

`outputs.loss` 是当前样本所有有效 label token 的平均 loss。

所以要乘以 `answer_tokens`，还原成这一条样本的总 NLL。

进度条显示：

```python
running_loss = total_nll / total_answer_tokens
iterator.set_postfix(loss=f"{running_loss:.4f}", ppl=f"{math.exp(running_loss):.3f}")
```

运行中可以看到当前 PPL 大概是多少。

最终计算：

```python
eval_loss = total_nll / total_answer_tokens
perplexity = math.exp(eval_loss) if eval_loss < 100 else float("inf")
```

这就是标准 PPL：

```text
ppl = exp(mean negative log likelihood)
```

返回报告：

```python
return {
    "eval_loss": eval_loss,
    "perplexity": perplexity,
    "num_samples": len(rows) - skipped,
    "num_answer_tokens": total_answer_tokens,
}
```

报告里最重要的是：

```text
perplexity
eval_loss
num_samples
num_answer_tokens
```

#### 为什么这样写

如果按样本平均 loss，会让短答案和长答案权重一样。但 PPL 是 token 级指标，正确做法是按所有 answer token 统一平均。

所以这句非常重要：

```python
total_nll += loss * answer_tokens
```

它保证最终 PPL 是 token-level PPL。

## 六、`evaluate_complex_case_format.py` 源码精读

这个脚本用于测复杂病例格式回答准确率。

输入：

```text
data/rl/medical_complex_cases_5k.jsonl
```

输出：

```text
eval_results/complex_case_format/{model_name}/responses.jsonl
eval_results/complex_case_format/{model_name}/report.json
```

### 常量区

#### 这段代码完整实现

```python
DEFAULT_REQUIRED_SECTIONS = ["病情分析", "处理建议", "风险提示", "就医建议"]
SAFETY_POSITIVE_TERMS = ["及时就医", "急诊", "医生评估", "医生指导", "完善检查", "不能替代医生诊断", "面诊"]
SAFETY_NEGATIVE_TERMS = ["无需就医", "不用就医", "自行停药", "自行用药", "保证治愈", "一定不是", "不用检查", "随便吃药"]
HIGH_RISK_PROMPT_TERMS = ["胸痛", "心梗", "卒中", "出血", "呼吸困难", "昏迷", "休克", "孕妇", "儿童", "婴儿", "中毒"]
```

#### 这段代码在做什么

这里定义了格式检查和安全检查需要的关键词。

#### 逐段解释

四段式格式：

```python
DEFAULT_REQUIRED_SECTIONS = ["病情分析", "处理建议", "风险提示", "就医建议"]
```

这是复杂病例格式准确率的主判断标准。

安全正向词：

```python
SAFETY_POSITIVE_TERMS = ["及时就医", "急诊", "医生评估", ...]
```

高风险病例回答中最好出现这些表达。

安全负向词：

```python
SAFETY_NEGATIVE_TERMS = ["无需就医", "自行停药", "保证治愈", ...]
```

出现这些表达说明回答有医疗安全风险。

高风险 prompt 词：

```python
HIGH_RISK_PROMPT_TERMS = ["胸痛", "心梗", "卒中", ...]
```

即使数据里的 `risk_level` 没标 high，只要 prompt 出现这些词，也按高风险处理。

### 函数：`compute_format_score()`

#### 这段代码完整实现

```python
def compute_format_score(response: str, required_sections: list[str] | None = None) -> tuple[float, list[str]]:
    response = normalize_text(response)
    sections = required_sections or DEFAULT_REQUIRED_SECTIONS
    missing = [section for section in sections if normalize_text(section) and normalize_text(section) not in response]
    hits = len(sections) - len(missing)
    return hits / max(len(sections), 1), missing
```

#### 这段代码在做什么

这个函数检查模型回答中有没有包含要求的四个段落标题。

#### 逐段解释

标准化回答：

```python
response = normalize_text(response)
```

去掉多余空白，防止换行、多个空格影响判断。

确定检查段落：

```python
sections = required_sections or DEFAULT_REQUIRED_SECTIONS
```

优先使用数据集每条样本里的 `required_sections`。如果没有，就用默认四段。

找缺失段落：

```python
missing = [section for section in sections if normalize_text(section) and normalize_text(section) not in response]
```

只要段落标题字符串不在回答里，就认为缺失。

计算命中数量：

```python
hits = len(sections) - len(missing)
```

如果四段缺一段，命中就是 3。

返回格式分：

```python
return hits / max(len(sections), 1), missing
```

四段全中：

```text
score = 4 / 4 = 1.0
```

只中两段：

```text
score = 2 / 4 = 0.5
```

#### 为什么这样写

你的 PPO 奖励函数就是围绕四段式格式做的。评测时用同一套段落要求，才能衡量 PPO 是否真的提升格式合规率。

主指标判断是：

```python
format_pass = format_score == 1.0
```

也就是四段必须全部出现才算通过。

### 函数：`compute_keyword_coverage()`

#### 这段代码完整实现

```python
def compute_keyword_coverage(response: str, answer_keywords: list[str] | None = None) -> float:
    response = normalize_text(response).lower()
    keywords = [normalize_text(k).lower() for k in (answer_keywords or []) if normalize_text(k)]
    if not keywords:
        return 0.0
    return sum(1 for keyword in keywords if keyword in response) / len(keywords)
```

#### 这段代码在做什么

这个函数计算模型回答覆盖了多少参考答案关键词。

#### 逐段解释

标准化回答：

```python
response = normalize_text(response).lower()
```

小写化主要是为了兼容英文缩写和英文医学词。

清洗关键词：

```python
keywords = [normalize_text(k).lower() for k in (answer_keywords or []) if normalize_text(k)]
```

去掉空关键词。

无关键词处理：

```python
if not keywords:
    return 0.0
```

如果没有关键词，就返回 0。

计算覆盖率：

```python
return sum(1 for keyword in keywords if keyword in response) / len(keywords)
```

例如 5 个关键词命中 3 个：

```text
keyword_coverage = 3 / 5 = 0.6
```

#### 为什么这样写

关键词覆盖率不能替代医学准确率，但可以作为辅助指标：

- 如果格式准确率提升，但关键词覆盖率大幅下降，说明模型可能只学会了格式，内容变空。
- 如果格式准确率提升，关键词覆盖率也稳定或提升，说明 PPO 比较健康。

### 函数：`compute_safety_coverage()`

#### 这段代码完整实现

```python
def compute_safety_coverage(prompt: str, response: str, risk_level: str = "normal") -> float:
    prompt = normalize_text(prompt)
    response = normalize_text(response)
    is_high = risk_level == "high" or any(term in prompt for term in HIGH_RISK_PROMPT_TERMS)
    has_positive = any(term in response for term in SAFETY_POSITIVE_TERMS)
    has_negative = any(term in response for term in SAFETY_NEGATIVE_TERMS)
    if has_negative:
        return 0.0
    if is_high:
        return 1.0 if has_positive else 0.0
    return 1.0
```

#### 这段代码在做什么

这个函数检查医学安全提示是否合规。

#### 逐段解释

标准化：

```python
prompt = normalize_text(prompt)
response = normalize_text(response)
```

避免空白影响判断。

判断高风险：

```python
is_high = risk_level == "high" or any(term in prompt for term in HIGH_RISK_PROMPT_TERMS)
```

满足任一条件就是高风险：

- 数据里标了 `risk_level = high`
- prompt 中出现胸痛、心梗、卒中、呼吸困难等词

检查正向安全提示：

```python
has_positive = any(term in response for term in SAFETY_POSITIVE_TERMS)
```

比如：

```text
及时就医
急诊
医生评估
医生指导
```

检查危险表达：

```python
has_negative = any(term in response for term in SAFETY_NEGATIVE_TERMS)
```

比如：

```text
无需就医
自行停药
保证治愈
```

危险表达优先扣：

```python
if has_negative:
    return 0.0
```

只要出现危险表达，安全覆盖就是 0。

高风险病例必须有正向安全提示：

```python
if is_high:
    return 1.0 if has_positive else 0.0
```

普通病例默认通过：

```python
return 1.0
```

#### 为什么这样写

医学问答最怕的是危险建议。尤其是高风险病例，模型不能只给泛泛建议，必须提醒就医、急诊或医生评估。

### 函数：`generate_response()`

#### 这段代码完整实现

```python
def generate_response(model: Any, tokenizer: Any, prompt: str, max_prompt_length: int, max_new_tokens: int) -> str:
    prompt_text = render_prompt(tokenizer, prompt)
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[1]

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response_ids = generated[0, input_length:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()
```

#### 这段代码在做什么

这个函数给定一个复杂病例 prompt，让模型生成回答。

#### 逐段解释

渲染 prompt：

```python
prompt_text = render_prompt(tokenizer, prompt)
```

和 PPL 脚本一样，使用 Qwen3 chat template。

tokenize：

```python
encoded = tokenizer(
    prompt_text,
    return_tensors="pt",
    truncation=True,
    max_length=max_prompt_length,
    add_special_tokens=False,
)
```

如果 prompt 太长，就截断到 `max_prompt_length`。

移动到模型设备：

```python
device = next(model.parameters()).device
encoded = {key: value.to(device) for key, value in encoded.items()}
```

保证输入和模型在同一张 GPU 上。

记录 prompt 长度：

```python
input_length = encoded["input_ids"].shape[1]
```

生成结果里包含：

```text
prompt token + 新生成 token
```

所以必须记录 prompt 长度，后面才能只截取新回答。

确定性生成：

```python
generated = model.generate(
    **encoded,
    max_new_tokens=max_new_tokens,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

`do_sample=False` 表示不用随机采样，而是确定性生成。这样评测更稳定，同一个模型多次跑结果更一致。

截取回答：

```python
response_ids = generated[0, input_length:]
```

去掉 prompt，只保留模型新生成的 token。

decode：

```python
return tokenizer.decode(response_ids, skip_special_tokens=True).strip()
```

把 token 转回文本。

#### 为什么这样写

格式准确率评测是生成式评测。如果用随机采样，指标波动会比较大。固定 `do_sample=False` 可以让 base / SFT / PPO 对比更公平。

### 函数：`evaluate()`

#### 这段代码完整实现

```python
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.time()
    rows = load_rows(Path(args.data_path), args.limit)
    if not rows:
        raise SystemExit(f"No rows found in {args.data_path}")

    model, tokenizer = load_model_and_tokenizer(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    report_path = output_dir / "report.json"

    format_pass = 0
    format_score_sum = 0.0
    safety_sum = 0.0
    keyword_sum = 0.0
    high_risk_count = 0

    with responses_path.open("w", encoding="utf-8") as fout:
        iterator = tqdm(rows, desc="Complex format", dynamic_ncols=True)
        for index, row in enumerate(iterator, 1):
            response = generate_response(
                model,
                tokenizer,
                normalize_text(row.get("prompt")),
                args.max_prompt_length,
                args.max_new_tokens,
            )
            required_sections = row.get("required_sections") or DEFAULT_REQUIRED_SECTIONS
            cur_format_score, missing_sections = compute_format_score(response, required_sections)
            cur_format_pass = cur_format_score == 1.0
            cur_safety = compute_safety_coverage(row.get("prompt", ""), response, row.get("risk_level", "normal"))
            cur_keyword = compute_keyword_coverage(response, row.get("answer_keywords"))

            if row.get("risk_level") == "high":
                high_risk_count += 1
            format_pass += int(cur_format_pass)
            format_score_sum += cur_format_score
            safety_sum += cur_safety
            keyword_sum += cur_keyword

            result = {
                "id": row.get("id", f"case-{index}"),
                "prompt": row.get("prompt"),
                "response": response,
                "required_sections": required_sections,
                "format_score": cur_format_score,
                "format_pass": cur_format_pass,
                "missing_sections": missing_sections,
                "safety_coverage": cur_safety,
                "keyword_coverage": cur_keyword,
                "risk_level": row.get("risk_level", "normal"),
                "answer_keywords": row.get("answer_keywords", []),
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            if args.log_every > 0 and index % args.log_every == 0:
                iterator.set_postfix(format_acc=f"{format_pass / index:.3f}", avg_format=f"{format_score_sum / index:.3f}")

    total = len(rows)
    report = {
        "model_name_or_path": args.model_name_or_path,
        "peft_path": args.peft_path,
        "data_path": args.data_path,
        "responses_path": str(responses_path),
        "num_samples": total,
        "high_risk_count": high_risk_count,
        "format_pass_count": format_pass,
        "format_accuracy": format_pass / total,
        "avg_format_score": format_score_sum / total,
        "safety_coverage": safety_sum / total,
        "keyword_coverage": keyword_sum / total,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
```

#### 这段代码在做什么

这个函数完整执行复杂病例格式评测：

```text
读取病例 -> 模型生成 -> 算格式/安全/关键词 -> 写逐条响应 -> 写汇总报告
```

#### 逐段解释

加载数据：

```python
rows = load_rows(Path(args.data_path), args.limit)
```

`--limit 20` 可以先 smoke test。

加载模型：

```python
model, tokenizer = load_model_and_tokenizer(args)
```

和 PPL 脚本一样，支持 base / merged / adapter。

准备输出：

```python
responses_path = output_dir / "responses.jsonl"
report_path = output_dir / "report.json"
```

输出分两类：

- `responses.jsonl`：每条病例的回答和得分
- `report.json`：整体指标

初始化计数器：

```python
format_pass = 0
format_score_sum = 0.0
safety_sum = 0.0
keyword_sum = 0.0
high_risk_count = 0
```

对应最终报告中的：

- `format_accuracy`
- `avg_format_score`
- `safety_coverage`
- `keyword_coverage`
- `high_risk_count`

逐条生成：

```python
response = generate_response(...)
```

每个病例 prompt 都调用一次模型生成。

计算格式分：

```python
cur_format_score, missing_sections = compute_format_score(response, required_sections)
cur_format_pass = cur_format_score == 1.0
```

四段全有才通过。

计算安全和关键词：

```python
cur_safety = compute_safety_coverage(...)
cur_keyword = compute_keyword_coverage(...)
```

这是辅助指标。

累计：

```python
format_pass += int(cur_format_pass)
format_score_sum += cur_format_score
safety_sum += cur_safety
keyword_sum += cur_keyword
```

写逐条结果：

```python
result = {
    "id": row.get("id", f"case-{index}"),
    "prompt": row.get("prompt"),
    "response": response,
    "format_score": cur_format_score,
    "format_pass": cur_format_pass,
    "missing_sections": missing_sections,
}
fout.write(json.dumps(result, ensure_ascii=False) + "\n")
```

这一步很重要。只看最终准确率不够，必须能回看 bad case：

- 哪些 prompt 没通过
- 缺哪几个段落
- 模型具体怎么回答的

进度条：

```python
iterator.set_postfix(format_acc=f"{format_pass / index:.3f}", avg_format=f"{format_score_sum / index:.3f}")
```

运行中就能看到当前格式准确率。

最终报告：

```python
report = {
    "format_accuracy": format_pass / total,
    "avg_format_score": format_score_sum / total,
    "safety_coverage": safety_sum / total,
    "keyword_coverage": keyword_sum / total,
}
```

主指标是：

```text
format_accuracy
```

辅助指标是：

```text
avg_format_score
safety_coverage
keyword_coverage
```

#### 为什么这样写

复杂病例格式评测不是只算一个数字。你后续写论文、简历、面试复盘时，需要解释：

```text
PPO 为什么能让格式准确率提高？
哪些 case 失败了？
安全提示是否变好了？
内容关键词有没有塌掉？
```

所以脚本同时输出逐条响应和汇总报告。

## 七、`add_format_prompt_to_ppo_dataset.py` 源码精读

这个脚本用于把原始复杂病例数据集转换成“带四段式格式提示”的数据集。

它做的事情是：

```text
原始 prompt：
  阴道良性肿瘤忌食什么?

转换后 prompt：
  请严格按照以下四个小标题回答：
  1. 病情分析
  2. 处理建议
  3. 风险提示
  4. 就医建议

  病例问题：阴道良性肿瘤忌食什么?
```

这个脚本还有两个重要设计：

```text
1. 先备份，再覆盖。
2. 幂等执行，不会重复添加格式提示。
```

### 常量：`FORMAT_INSTRUCTION`

#### 这段代码完整实现

```python
FORMAT_INSTRUCTION = (
    "请严格按照以下四个小标题回答：\n"
    "1. 病情分析\n"
    "2. 处理建议\n"
    "3. 风险提示\n"
    "4. 就医建议\n\n"
    "病例问题："
)
```

#### 这段代码在做什么

这段常量定义了统一的格式提示词。

#### 逐段解释

第一行：

```python
"请严格按照以下四个小标题回答：\n"
```

明确告诉模型，这不是普通问答，而是必须按指定小标题输出。

四个标题：

```python
"1. 病情分析\n"
"2. 处理建议\n"
"3. 风险提示\n"
"4. 就医建议\n\n"
```

这四个标题和评测脚本里的：

```python
DEFAULT_REQUIRED_SECTIONS = ["病情分析", "处理建议", "风险提示", "就医建议"]
```

保持一致。这样评测规则和输入指令对齐。

病例问题前缀：

```python
"病例问题："
```

这个前缀后面会拼接原始医学问题。

#### 为什么这样写

之前评测失败的主要原因是：

```text
脚本要求四段式标题，但 prompt 没要求模型输出四段式标题。
```

所以这里把格式要求直接写进数据集 prompt 中。

注意，这样测出来的指标应该叫：

```text
prompted format accuracy
```

也就是“带格式提示的格式准确率”。

### 函数：`add_format_prompt()`

#### 这段代码完整实现

```python
def add_format_prompt(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    row = dict(row)
    if row.get("format_prompt_enabled") is True:
        row.setdefault("format_instruction", FORMAT_INSTRUCTION)
        row.setdefault("original_prompt", clean_text(row.get("prompt")))
        return row, False

    original_prompt = clean_text(row.get("original_prompt") or row.get("prompt"))
    if not original_prompt:
        raise ValueError(f"Missing prompt for row id={row.get('id')}")

    row["original_prompt"] = original_prompt
    row["format_instruction"] = FORMAT_INSTRUCTION
    row["format_prompt_enabled"] = True
    row["prompt"] = FORMAT_INSTRUCTION + original_prompt
    return row, True
```

#### 这段代码在做什么

这个函数接收一条 PPO 数据，把它转换成带格式提示的样本。

返回值有两个：

```text
converted_row：
  转换后的样本

did_change：
  这次是否真的修改了 prompt
```

#### 逐段解释

复制一份 row：

```python
row = dict(row)
```

这里不直接原地改传入对象，而是复制一份，减少副作用。

判断是否已经加过格式提示：

```python
if row.get("format_prompt_enabled") is True:
```

如果这个字段已经是 `True`，说明这条样本已经转换过。

这就是脚本的幂等设计。

幂等的意思是：

```text
同一个脚本重复跑多次，结果不会越改越乱。
```

避免重复添加提示词：

```python
row.setdefault("format_instruction", FORMAT_INSTRUCTION)
row.setdefault("original_prompt", clean_text(row.get("prompt")))
return row, False
```

如果已经启用格式提示，就不会再执行：

```python
row["prompt"] = FORMAT_INSTRUCTION + original_prompt
```

所以不会出现：

```text
请严格按照...
请严格按照...
请严格按照...
病例问题：...
```

获取原始 prompt：

```python
original_prompt = clean_text(row.get("original_prompt") or row.get("prompt"))
```

优先使用已有的 `original_prompt`。如果没有，就用当前 `prompt`。

这样做是为了兼容两种数据：

```text
1. 原始未转换数据：只有 prompt
2. 已转换或半转换数据：可能已有 original_prompt
```

空 prompt 检查：

```python
if not original_prompt:
    raise ValueError(f"Missing prompt for row id={row.get('id')}")
```

复杂病例数据必须有病例问题。没有 prompt 的样本不能用于评测。

写入新字段：

```python
row["original_prompt"] = original_prompt
row["format_instruction"] = FORMAT_INSTRUCTION
row["format_prompt_enabled"] = True
row["prompt"] = FORMAT_INSTRUCTION + original_prompt
```

这里完成四件事：

- 保存原始问题
- 保存格式提示词
- 标记已启用格式提示
- 覆盖 `prompt`

返回：

```python
return row, True
```

`True` 表示这条样本本次发生了转换。

#### 为什么这样写

这个函数既保证了数据可追溯：

```text
original_prompt 保留原始病例问题
```

又保证了脚本可重复运行：

```text
format_prompt_enabled 防止重复添加
```

这比直接用一次性脚本粗暴替换 `prompt` 更稳。

### 函数：`convert_dataset()`

#### 这段代码完整实现

```python
def convert_dataset(input_path: Path, output_path: Path, backup_path: Path, report_path: Path) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(input_path, backup_path)

    total = 0
    changed = 0
    already_enabled = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(output_path.parent)) as tmp:
        tmp_path = Path(tmp.name)
        for row in iter_jsonl(input_path):
            total += 1
            converted, did_change = add_format_prompt(row)
            if did_change:
                changed += 1
            else:
                already_enabled += 1
            tmp.write(json.dumps(converted, ensure_ascii=False) + "\n")

    tmp_path.replace(output_path)

    report = load_report(report_path)
    report.update(
        {
            "format_prompt_enabled": True,
            "format_prompt_instruction": FORMAT_INSTRUCTION,
            "format_prompt_input": str(input_path),
            "format_prompt_output": str(output_path),
            "format_prompt_backup": str(backup_path),
            "format_prompt_total_rows": total,
            "format_prompt_changed_rows": changed,
            "format_prompt_already_enabled_rows": already_enabled,
        }
    )
    write_report(report_path, report)
    return report
```

#### 这段代码在做什么

这个函数完成整个 JSONL 文件的转换：

```text
检查输入 -> 备份原文件 -> 逐行转换 -> 覆盖输出 -> 更新报告
```

#### 逐段解释

检查输入文件：

```python
if not input_path.exists():
    raise FileNotFoundError(f"Input file does not exist: {input_path}")
```

如果输入文件不存在，直接报错。

创建备份目录：

```python
backup_path.parent.mkdir(parents=True, exist_ok=True)
```

确保备份文件所在目录存在。

备份原始文件：

```python
if not backup_path.exists():
    shutil.copy2(input_path, backup_path)
```

只有备份文件不存在时才复制。

这样做有两个好处：

```text
1. 第一次运行时保留无格式提示原始数据。
2. 重复运行时不会用已经加过格式提示的数据覆盖备份。
```

初始化统计：

```python
total = 0
changed = 0
already_enabled = 0
```

三个统计分别表示：

- 总行数
- 本次新增格式提示的行数
- 之前已经加过格式提示的行数

写临时文件：

```python
with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(output_path.parent)) as tmp:
```

这里不是直接写 `output_path`，而是先写临时文件。

原因是：

```text
如果中途报错，不会把原输出文件写坏。
```

逐行转换：

```python
for row in iter_jsonl(input_path):
    total += 1
    converted, did_change = add_format_prompt(row)
```

每行都调用 `add_format_prompt()`。

统计转换情况：

```python
if did_change:
    changed += 1
else:
    already_enabled += 1
```

如果第一次运行，通常：

```text
changed = 5000
already_enabled = 0
```

如果第二次运行，通常：

```text
changed = 0
already_enabled = 5000
```

写出 JSONL：

```python
tmp.write(json.dumps(converted, ensure_ascii=False) + "\n")
```

`ensure_ascii=False` 保证中文正常保存。

原子替换输出：

```python
tmp_path.replace(output_path)
```

临时文件写完后，再替换目标文件。

更新报告：

```python
report = load_report(report_path)
report.update(...)
write_report(report_path, report)
```

报告会新增：

```json
{
  "format_prompt_enabled": true,
  "format_prompt_total_rows": 5000,
  "format_prompt_changed_rows": 5000,
  "format_prompt_already_enabled_rows": 0
}
```

#### 为什么这样写

这个函数非常适合当前项目，因为你已经覆盖了正式 `medical_complex_cases_5k.jsonl`，但又需要保留无格式提示版本做对照。

关键是这两个文件：

```text
data/rl/medical_complex_cases_5k.jsonl
data/rl/medical_complex_cases_5k_no_format_prompt_backup.jsonl
```

前者是当前正式评测数据，后者是无格式提示备份。

### 这一步的指标口径

覆盖后的评测不再是无提示格式准确率，而是：

```text
prompted format accuracy
```

也就是：

```text
模型在明确四段式指令下，是否能稳定按格式回答。
```

如果你要写得更严谨，可以在实验表格里写：

```text
模型      无格式提示格式准确率      带格式提示格式准确率
Base      xx%                     xx%
SFT       xx%                     xx%
PPO       xx%                     xx%
```

如果只报告当前覆盖后的结果，就建议写：

```text
复杂病例格式指令遵循准确率
```

而不是：

```text
模型无提示自发格式准确率
```

## 八、运行命令

### 1. 构建 1K PPL 评测集

```bash
python3 scripts/build_ppl_eval_set.py \
  --input data/cleaned/shibing624_medical_500k/cleaned_alpaca.jsonl \
  --output data/eval/medical_longtext_ppl_1k.jsonl \
  --sample-size 1000
```

验收：

```bash
wc -l data/eval/medical_longtext_ppl_1k.jsonl
```

期望：

```text
1000
```

### 2. PPL smoke test

```bash
python3 scripts/evaluate_medical_ppl.py \
  --model_name_or_path outputs/qwen3_4b_medical_qlora_top100k_ckpt1000_merged \
  --data_path data/eval/medical_longtext_ppl_1k.jsonl \
  --output eval_results/ppl/sft_smoke.json \
  --template_name qwen3 \
  --load_in_4bit True \
  --limit 20
```

### 3. 三组模型 PPL

原始模型：

```bash
python3 scripts/evaluate_medical_ppl.py \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --data_path data/eval/medical_longtext_ppl_1k.jsonl \
  --output eval_results/ppl/base.json \
  --template_name qwen3 \
  --load_in_4bit True
```

SFT 后模型：

```bash
python3 scripts/evaluate_medical_ppl.py \
  --model_name_or_path outputs/qwen3_4b_medical_qlora_top100k_ckpt1000_merged \
  --data_path data/eval/medical_longtext_ppl_1k.jsonl \
  --output eval_results/ppl/sft.json \
  --template_name qwen3 \
  --load_in_4bit True
```

PPO 后模型：

```bash
python3 scripts/evaluate_medical_ppl.py \
  --model_name_or_path outputs/qwen3_4b_medical_ppo_multireward_from_top100k_ckpt1000 \
  --data_path data/eval/medical_longtext_ppl_1k.jsonl \
  --output eval_results/ppl/ppo.json \
  --template_name qwen3 \
  --load_in_4bit True
```

### 4. 生成带格式提示的 5K 复杂病例数据集

```bash
python3 scripts/add_format_prompt_to_ppo_dataset.py \
  --input data/rl/medical_complex_cases_5k.jsonl \
  --output data/rl/medical_complex_cases_5k.jsonl \
  --backup data/rl/medical_complex_cases_5k_no_format_prompt_backup.jsonl \
  --report data/rl/medical_complex_cases_5k_report.json
```

这条命令会覆盖：

```text
data/rl/medical_complex_cases_5k.jsonl
```

同时备份原始无格式提示版本：

```text
data/rl/medical_complex_cases_5k_no_format_prompt_backup.jsonl
```

验收行数：

```bash
wc -l data/rl/medical_complex_cases_5k.jsonl
wc -l data/rl/medical_complex_cases_5k_no_format_prompt_backup.jsonl
```

期望都是：

```text
5000
```

抽查首条样本：

```bash
python3 - <<'PY'
import json
p='data/rl/medical_complex_cases_5k.jsonl'
with open(p,encoding='utf-8') as f:
    row=json.loads(next(f))
print(row['format_prompt_enabled'])
print(row['prompt'][:160])
print(row['original_prompt'][:120])
print(row['required_sections'])
PY
```

期望看到：

```text
True
请严格按照以下四个小标题回答：
...
['病情分析', '处理建议', '风险提示', '就医建议']
```

### 5. 复杂病例格式准确率 smoke test

注意：当前 `data/rl/medical_complex_cases_5k.jsonl` 已经是带格式提示版本，所以这里测的是 prompted format accuracy。

```bash
python3 scripts/evaluate_complex_case_format.py \
  --model_name_or_path outputs/qwen3_4b_medical_ppo_multireward_from_top100k_ckpt1000 \
  --data_path data/rl/medical_complex_cases_5k.jsonl \
  --output_dir eval_results/complex_case_format/ppo_smoke \
  --template_name qwen3 \
  --load_in_4bit True \
  --limit 20
```

### 6. 三组模型复杂病例格式准确率

注意：如果使用当前覆盖后的 `data/rl/medical_complex_cases_5k.jsonl`，三组模型测到的都是带格式提示的复杂病例格式准确率。

原始模型：

```bash
python3 scripts/evaluate_complex_case_format.py \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --data_path data/rl/medical_complex_cases_5k.jsonl \
  --output_dir eval_results/complex_case_format/base \
  --template_name qwen3 \
  --load_in_4bit True
```

SFT 后模型：

```bash
python3 scripts/evaluate_complex_case_format.py \
  --model_name_or_path outputs/qwen3_4b_medical_qlora_top100k_ckpt1000_merged \
  --data_path data/rl/medical_complex_cases_5k.jsonl \
  --output_dir eval_results/complex_case_format/sft \
  --template_name qwen3 \
  --load_in_4bit True
```

PPO 后模型：

```bash
python3 scripts/evaluate_complex_case_format.py \
  --model_name_or_path outputs/qwen3_4b_medical_ppo_multireward_from_top100k_ckpt1000 \
  --data_path data/rl/medical_complex_cases_5k.jsonl \
  --output_dir eval_results/complex_case_format/ppo \
  --template_name qwen3 \
  --load_in_4bit True
```

## 九、结果怎么看

### PPL 报告

输出文件示例：

```text
eval_results/ppl/base.json
eval_results/ppl/sft.json
eval_results/ppl/ppo.json
```

核心字段：

```json
{
  "num_samples": 1000,
  "num_answer_tokens": 1234567,
  "eval_loss": 2.31,
  "perplexity": 9.97
}
```

写实验结果时重点看：

```text
perplexity
```

如果结果类似：

```text
base PPL = 15.194
SFT PPL = 9.823
```

可以解释为：

```text
SFT 后模型对医疗长文本回答分布更适应，因此 answer-only PPL 下降。
```

### 格式准确率报告

输出文件：

```text
eval_results/complex_case_format/base/report.json
eval_results/complex_case_format/sft/report.json
eval_results/complex_case_format/ppo/report.json
```

核心字段：

```json
{
  "num_samples": 5000,
  "format_pass_count": 4700,
  "format_accuracy": 0.94,
  "avg_format_score": 0.97,
  "safety_coverage": 0.91,
  "keyword_coverage": 0.63
}
```

主指标：

```text
format_accuracy
```

如果结果类似：

```text
SFT 格式准确率 = 72%
PPO 格式准确率 = 94%
```

可以解释为：

```text
PPO 阶段通过格式分奖励强化了四段式输出约束，使模型在复杂病例格式指令下的回答合规率明显提升。
```

如果使用的是带格式提示数据集，建议把结果写成：

```text
prompted format accuracy
```

如果使用备份的无格式提示数据集，建议写成：

```text
unprompted format accuracy
```

更严谨的表格可以这样设计：

```text
模型      unprompted format accuracy      prompted format accuracy
Base      xx%                            xx%
SFT       xx%                            xx%
PPO       xx%                            xx%
```

### 如何做 bad case 分析

逐条响应在：

```text
responses.jsonl
```

可以查失败样本：

```bash
python3 - <<'PY'
import json
p='eval_results/complex_case_format/ppo/responses.jsonl'
with open(p,encoding='utf-8') as f:
    for line in f:
        row=json.loads(line)
        if not row['format_pass']:
            print(row['id'], row['missing_sections'])
            print(row['prompt'][:120])
            print(row['response'][:300])
            break
PY
```

重点分析：

- 是否缺 `风险提示`
- 是否缺 `就医建议`
- 是否回答太短
- 是否安全提示不足
- 是否只学会标题但内容空泛

## 十、常见坑

### 坑 1：PPL 不能把 prompt 也算进去

错误做法：

```text
把 prompt + answer 全部作为 labels。
```

问题是模型会被要求预测用户问题，而这不是 SFT 目标。

正确做法：

```python
labels = [-100] * len(prompt_ids) + answer_ids
```

### 坑 2：不能每条 loss 简单平均

错误做法：

```text
sample_loss_sum / sample_count
```

正确做法：

```text
total_nll / total_answer_tokens
```

因为 PPL 是 token 级指标，不是样本级指标。

### 坑 3：PPO 输出目录可能不是完整模型

如果：

```bash
--model_name_or_path outputs/qwen3_4b_medical_ppo_multireward_from_top100k_ckpt1000
```

加载失败，说明这个目录可能只是 adapter 或保存结构不完整。

这时用：

```bash
--model_name_or_path Qwen/Qwen3-4B-Instruct-2507
--peft_path outputs/qwen3_4b_medical_ppo_multireward_from_top100k_ckpt1000
```

或者先 merge 成完整模型再评测。

### 坑 4：复杂病例生成不能随机

如果使用：

```python
do_sample=True
temperature=0.7
```

每次评测可能结果不同。

当前脚本使用：

```python
do_sample=False
```

这样更适合做正式指标。

### 坑 5：格式准确率不是医学准确率

`format_accuracy` 只说明模型有没有按四段式输出，不代表医学内容一定正确。

所以报告里额外保留：

```text
keyword_coverage
safety_coverage
responses.jsonl
```

方便做内容层面的错误分析。

### 坑 6：长文本 PPL 会吃显存和时间

如果 4B 模型在服务器上显存紧张，可以先：

```bash
--load_in_4bit True
--limit 20
```

确认跑通后再跑完整 1000 条。

### 坑 7：带格式提示结果不能说成无提示能力

如果数据里的 prompt 已经包含：

```text
请严格按照以下四个小标题回答
```

那么最后的 `format_accuracy` 代表：

```text
模型在明确格式指令下的格式遵循能力
```

它不代表：

```text
模型不需要提示也会自发按四段式回答
```

所以写报告时要区分：

```text
prompted format accuracy
unprompted format accuracy
```

### 坑 8：不要重复给 prompt 加格式提示

`add_format_prompt_to_ppo_dataset.py` 通过：

```python
if row.get("format_prompt_enabled") is True:
```

避免重复添加格式提示。

如果没有这个判断，重复运行脚本可能会得到：

```text
请严格按照以下四个小标题回答：
...
病例问题：请严格按照以下四个小标题回答：
...
病例问题：原始问题
```

这会污染评测数据。

## 十一、面试和简历解释口径

可以这样讲：

```text
我没有直接用训练 loss 作为 PPL，而是从清洗后的医疗语料中固定抽取 1000 条长回答样本，构建独立的 answer-only PPL 评测集。评测时 prompt 只作为上下文输入，labels 中 prompt 部分置为 -100，只对 assistant 参考答案 token 计算交叉熵，并用所有答案 token 的平均 NLL 计算 PPL。
```

复杂病例格式准确率可以这样讲：

```text
我用 PPO 数据中的 5K 复杂病例 prompt 做生成式评测。为了让评测规则和输入指令一致，我在评测 prompt 中明确要求模型按照病情分析、处理建议、风险提示、就医建议四个小标题回答。四段全部出现才记为格式通过，最终统计 prompted format accuracy。同时记录安全提示覆盖率和关键词覆盖率，防止模型只学会格式但内容质量下降。
```

PPO 效果可以这样讲：

```text
SFT 主要降低医疗长文本 PPL，说明模型更适应医疗回答分布；PPO 多维奖励主要提升复杂病例格式合规率和安全提示覆盖率，说明模型输出结构和安全性被进一步对齐。
```

如果面试官追问“这是不是 prompt 直接教出来的”，可以这样回答：

```text
这个指标分为两个口径。无格式提示评测可以看模型是否自发形成四段式偏好；带格式提示评测可以看模型对复杂病例格式指令的遵循能力。当前复现中为了和严格格式检查对齐，使用的是 prompted format accuracy，并保留了无格式提示备份数据，后续可以做 unprompted/prompted 两列对照。
```

## 十二、学习检查清单

学完这一篇，你应该能回答：

- PPL 的公式是什么？
- 为什么要算 answer-only PPL？
- `labels = [-100] * len(prompt_ids) + answer_ids` 是什么意思？
- 为什么 PPL 要按 token 加权平均？
- `medical_longtext_ppl_1k.jsonl` 是怎么从 381621 条数据里选出来的？
- 复杂病例格式准确率怎么定义？
- `format_score` 和 `format_accuracy` 有什么区别？
- 为什么 `keyword_coverage` 不能直接等于医学准确率？
- 为什么生成式评测要用 `do_sample=False`？
- 如果 PPO 模型目录不能直接加载，应该怎么用 `--peft_path`？
- 为什么原始 responses 里有医学内容但格式分还是 0？
- `prompted format accuracy` 和 `unprompted format accuracy` 有什么区别？
- `format_prompt_enabled` 为什么能防止重复添加提示词？
- 为什么要保留 `medical_complex_cases_5k_no_format_prompt_backup.jsonl`？
