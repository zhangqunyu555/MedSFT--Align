# 02 C-Eval 医学双目标集构建

## 1. 本阶段目标

本阶段把 C-Eval 的两个医学科目下载并整理为两个独立 JSONL 数据集：

- `clinical_medicine`：临床医学
- `basic_medicine`：基础医学

两个 JSONL 都包含同一批题目，并且每条样本使用相同 `id` 对齐：

```text
data/eval/ceval_medical_question_only.jsonl
data/eval/ceval_medical_question_with_answer.jsonl
```

区别是：

- `question_only`：不带正确答案，只用题干作为目标域语义。
- `question_with_answer`：把正确答案拼进题干，用题干加正确医学知识点作为目标域语义。

脚本位置：

```bash
scripts/build_ceval_medical_jsonl.py
```

报告文件：

```bash
data/eval/ceval_medical_target_report.json
```

`data/eval/` 已在 `.gitignore` 中忽略，因此真实下载数据不会上传 Git。

## 2. 为什么要拆成两个数据集

向量筛选阶段可以有两个实验口径：

1. 只按题干语义筛选：更接近真实评测输入，因为模型评测时只看到题目和选项，不应该看到答案。
2. 按题干加正确答案筛选：更聚焦正确医学知识点，有助于从 50 万条医疗语料中找到知识更匹配的样本。

把它们拆成两个 JSONL，而不是放在同一个文件里，有三个好处：

- 后续 embedding 时可以分别编码，避免混淆。
- 可以比较两种目标集筛出来的 Top-100k 数据有什么差异。
- 训练污染边界更清楚：带答案目标集只能用于相似度筛选实验，不能直接混入 SFT 训练。

## 3. 两个 JSONL 的格式

### 3.1 不带答案数据集

文件：

```bash
data/eval/ceval_medical_question_only.jsonl
```

每行格式：

```json
{
  "id": "clinical_medicine-test-0",
  "source": "ceval/ceval-exam",
  "subject": "clinical_medicine",
  "subject_zh": "临床医学",
  "split": "test",
  "question": "原始题干",
  "options": {
    "A": "选项A",
    "B": "选项B",
    "C": "选项C",
    "D": "选项D"
  },
  "target_text": "科目：临床医学\n题目：原始题干"
}
```

这个文件不包含 `answer`、`answer_text`、`question_with_answer`。

### 3.2 带答案数据集

文件：

```bash
data/eval/ceval_medical_question_with_answer.jsonl
```

每行格式：

```json
{
  "id": "clinical_medicine-test-0",
  "source": "ceval/ceval-exam",
  "subject": "clinical_medicine",
  "subject_zh": "临床医学",
  "split": "test",
  "question": "原始题干",
  "options": {
    "A": "选项A",
    "B": "选项B",
    "C": "选项C",
    "D": "选项D"
  },
  "answer": "C",
  "answer_text": "正确选项文本",
  "explanation": "",
  "question_with_answer": "原始题干\n正确答案：C. 正确选项文本",
  "target_text": "科目：临床医学\n题目：原始题干\n正确答案：C. 正确选项文本"
}
```

这个文件把正确答案放在 `question_with_answer` 和 `target_text` 中。

## 4. 运行命令

正式下载两个医学科目的 `dev`、`val`、`test`：

```bash
python3 scripts/build_ceval_medical_jsonl.py
```

等价于：

```bash
python3 scripts/build_ceval_medical_jsonl.py \
  --question-only-output data/eval/ceval_medical_question_only.jsonl \
  --question-answer-output data/eval/ceval_medical_question_with_answer.jsonl \
  --report data/eval/ceval_medical_target_report.json
```

小规模测试，只取 `val` 中每个科目前 2 条：

```bash
python3 scripts/build_ceval_medical_jsonl.py \
  --splits val \
  --limit-per-split 2 \
  --question-only-output /private/tmp/ceval_question_only.jsonl \
  --question-answer-output /private/tmp/ceval_question_with_answer.jsonl \
  --report /private/tmp/ceval_target_report.json
```

## 5. 关键函数学习

### 5.1 `parse_args()`：定义两个输出文件

关键语句：

```python
parser.add_argument("--question-only-output", default="data/eval/ceval_medical_question_only.jsonl")
parser.add_argument("--question-answer-output", default="data/eval/ceval_medical_question_with_answer.jsonl")
parser.add_argument("--report", default="data/eval/ceval_medical_target_report.json")
```

含义：

- `--question-only-output` 是不带答案的数据集。
- `--question-answer-output` 是带答案的数据集。
- `--report` 是下载和转换报告。

### 5.2 `fetch_rows()`：调用 Hugging Face Dataset Viewer API

关键语句：

```python
query = urlencode({
    "dataset": DATASET,
    "config": subject,
    "split": split,
    "offset": offset,
    "length": length,
})
```

它会拼出类似请求：

```text
https://datasets-server.huggingface.co/rows?dataset=ceval/ceval-exam&config=clinical_medicine&split=val&offset=0&length=100
```

关键语句：

```python
with urlopen(request, timeout=timeout) as response:
    return json.loads(response.read().decode("utf-8"))
```

这句话完成下载，并把返回 JSON 解析成 Python 字典。

### 5.3 `fetch_rows_with_retries()`：网络失败重试

下载公开数据时可能遇到 SSL 断连、临时超时，或者远端直接断开连接。脚本会重试：

```python
for attempt in range(retries + 1):
```

如果 `--retries 3`，就是初始请求 1 次，加最多 3 次重试。

退避等待：

```python
time.sleep(min(2**attempt, 8))
```

连续失败时会逐步等更久，但最多等 8 秒。

### 5.4 `iter_dataset_rows()`：分页下载

Dataset Viewer API 需要分页读取：

```python
offset += len(page_rows)
```

停止条件：

```python
if total is not None and offset >= int(total):
    break
if len(page_rows) < length:
    break
```

这样可以稳定拉取一个科目某个 split 的全部题目。

### 5.5 `convert_row()`：一行 C-Eval 生成两个目标记录

这个函数是本阶段核心。

先取题干：

```python
question = clean_text(row.get("question"))
if not question:
    return None, None, "missing_question"
```

再取四个选项：

```python
options = {choice: clean_text(row.get(choice)) for choice in CHOICES}
if any(not options[choice] for choice in CHOICES):
    return None, None, "missing_option"
```

再取正确答案：

```python
answer = clean_text(row.get("answer")).upper()
if not answer:
    return None, None, "missing_answer"
if answer not in options:
    return None, None, "invalid_answer"
```

正确答案映射：

```python
answer_text = options[answer]
```

如果 `answer = "B"`，那 `answer_text = options["B"]`。

然后构造两个记录：

```python
question_only_record = {
    **base_record,
    "target_text": question_only_target_text,
}
```

和：

```python
question_answer_record = {
    **base_record,
    "answer": answer,
    "answer_text": answer_text,
    "question_with_answer": question_with_answer,
    "target_text": question_answer_target_text,
}
```

两个记录共享同一个 `base_record`，所以 `id`、`subject`、`split`、`question`、`options` 都一致。

### 5.6 `build_question_only_target_text()`：不带答案目标文本

关键语句：

```python
return f"科目：{subject_zh}\n题目：{question}"
```

这个文本用于“只按题干语义”的向量筛选。

### 5.7 `build_question_with_answer()`：把正确答案放进题干

关键语句：

```python
return f"{question}\n正确答案：{answer}. {answer_text}"
```

这就是“带答案数据集”里 `question_with_answer` 的来源。

### 5.8 `build_question_answer_target_text()`：带答案目标文本

关键语句：

```python
return f"科目：{subject_zh}\n题目：{question}\n正确答案：{answer}. {answer_text}"
```

这个文本用于“题干 + 正确医学知识点”的向量筛选。

### 5.9 `build_dataset()`：两个文件同时写出

关键语句：

```python
with question_only_path.open("w", encoding="utf-8") as question_only_file, question_answer_path.open(
    "w", encoding="utf-8"
) as question_answer_file:
```

脚本同时打开两个输出文件。

写出逻辑：

```python
write_jsonl_line(question_only_file, question_only_record)
write_jsonl_line(question_answer_file, question_answer_record)
```

同一条 C-Eval 原始题会连续写入两个文件，因此两个文件行数和 `id` 顺序保持一致。

## 6. 报告文件怎么读

报告文件默认是：

```bash
data/eval/ceval_medical_target_report.json
```

重要字段：

- `stats.kept`：两个文件共同保留的题目数。
- `question_only_output`：不带答案 JSONL 路径。
- `question_answer_output`：带答案 JSONL 路径。
- `per_subject_split`：每个科目和 split 的下载统计。
- `errors`：下载失败列表。

## 7. 验收标准

本阶段完成后应满足：

- 有两个 JSONL 文件，而不是一个混合文件。
- 两个 JSONL 行数一致。
- 两个 JSONL 的 `id` 顺序完全一致。
- 不带答案文件的 `target_text` 不包含 `正确答案：`。
- 带答案文件的 `target_text` 和 `question_with_answer` 包含 `正确答案：`。
- 带答案文件满足 `answer_text == options[answer]`。
- 两个文件都在 `data/eval/` 下，被 `.gitignore` 忽略。

验证命令：

```bash
python3 -m py_compile scripts/build_ceval_medical_jsonl.py

python3 scripts/build_ceval_medical_jsonl.py \
  --splits val \
  --limit-per-split 2 \
  --question-only-output /private/tmp/ceval_question_only.jsonl \
  --question-answer-output /private/tmp/ceval_question_with_answer.jsonl \
  --report /private/tmp/ceval_target_report.json
```

## 8. 与训练污染的边界

不带答案数据集可以用于更接近评测输入的目标域筛选实验。

带答案数据集只能用于“目标域知识点相似度筛选”实验，不能直接作为 SFT 训练样本，也不能把其中正确答案蒸馏进训练集。

旧版文件：

```bash
data/eval/ceval_medical_target.jsonl
```

不再作为推荐输入。推荐使用新的两个文件：

```bash
data/eval/ceval_medical_question_only.jsonl
data/eval/ceval_medical_question_with_answer.jsonl
```
