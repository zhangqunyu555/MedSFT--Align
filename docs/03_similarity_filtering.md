# 03 向量相似筛选学习文档

## 1. 本阶段目标

本阶段要把清洗后的约 50 万条医疗 SFT 样本，与 C-Eval 医学目标集做向量相似度计算，筛出得分最高的 10 万条数据。

脚本位置：

```bash
scripts/filter_by_ceval_similarity.py
```

默认配置：

```bash
configs/similarity_filtering.yaml
```

默认输入：

```bash
data/cleaned/cleaned_alpaca.jsonl
data/eval/ceval_medical_question_only.jsonl
```

默认输出：

```bash
data/sft/medical_sft_top100k_by_ceval_similarity.jsonl
data/sft/medical_sft_top100k_by_ceval_similarity_report.json
```

`data/sft/` 已经被 `.gitignore` 忽略，因此筛出来的真实训练数据不会上传 Git。

## 2. 核心思路

向量相似筛选的关键是把文本变成向量，然后用余弦相似度比较语义接近程度。

流程：

```text
读取 C-Eval 医学目标集
  -> 编码 target_text，得到 target_embeddings
  -> 逐批读取 50 万 SFT 样本
  -> 拼接 instruction/input/output 为候选文本
  -> 编码候选文本，得到 candidate_embeddings
  -> 矩阵乘法计算相似度
  -> 每条候选取最相似的 C-Eval 题目分数
  -> 用最小堆保留 Top-100k
  -> 写出筛选后的 SFT JSONL 和报告
```

如果向量已经 L2 归一化，余弦相似度就等价于点积：

```python
scores = candidate_embeddings @ target_embeddings.T
```

每条候选样本取它与所有 C-Eval 目标题的最高分：

```python
best_scores, best_indices = score_matrix.max(dim=1)
```

## 3. 正式运行

默认使用不带答案的 C-Eval 目标集：

```bash
python3 scripts/filter_by_ceval_similarity.py \
  --config configs/similarity_filtering.yaml \
  --input data/cleaned/cleaned_alpaca.jsonl \
  --target data/eval/ceval_medical_question_only.jsonl \
  --output data/sft/medical_sft_top100k_by_ceval_similarity.jsonl \
  --top-k 100000
```

使用带答案目标集做对照实验：

```bash
python3 scripts/filter_by_ceval_similarity.py \
  --config configs/similarity_filtering.yaml \
  --target data/eval/ceval_medical_question_with_answer.jsonl \
  --output data/sft/medical_sft_top100k_by_ceval_similarity_with_answer.jsonl \
  --report data/sft/medical_sft_top100k_by_ceval_similarity_with_answer_report.json \
  --top-k 100000
```

## 4. 配置说明

核心配置在 [configs/similarity_filtering.yaml](../configs/similarity_filtering.yaml)：

```yaml
embedding_backend: transformers
embedding_model: BAAI/bge-small-zh-v1.5
device: auto
batch_size: 32
target_batch_size: 64
max_length: 512
top_k: 100000
log_every: 500
```

含义：

- `embedding_backend: transformers`：正式使用 Hugging Face Transformers 模型编码。
- `embedding_model`：默认使用 `BAAI/bge-small-zh-v1.5`。
- `device: auto`：自动选择 `cuda`、`mps` 或 `cpu`。
- `batch_size`：50 万候选样本的编码 batch。
- `target_batch_size`：C-Eval 目标集的编码 batch。
- `top_k`：最终保留条数。
- `log_every`：每隔多少个 batch 输出一次进度。

本地逻辑测试可以临时使用：

```bash
--embedding-backend hash
```

`hash` 后端不是真实语义向量，只用于验证 Top-K、字段和报告逻辑，不用于正式筛选。

## 5. 关键函数学习

### 5.1 `parse_args()`：命令行入口

关键语句：

```python
parser.add_argument("--input", dest="input_path", default=None)
parser.add_argument("--target", dest="target_path", default=None)
parser.add_argument("--output", dest="output_path", default=None)
parser.add_argument("--top-k", type=int, default=None)
```

这些参数让你可以在不改配置文件的情况下切换输入、目标集、输出和 Top-K。

### 5.2 `load_config()` 和 `apply_cli_overrides()`：配置优先级

先读取 YAML：

```python
config = apply_cli_overrides(load_config(args.config), args)
```

命令行参数优先级高于配置文件。比如配置里是 `top_k: 100000`，但命令行传 `--top-k 3`，最终会用 3。

### 5.3 `TransformersEmbedder`：真实 embedding 编码器

加载模型：

```python
self.tokenizer = AutoTokenizer.from_pretrained(model_name)
self.model = AutoModel.from_pretrained(model_name)
```

前向计算：

```python
output = self.model(**encoded)
token_embeddings = output.last_hidden_state
```

平均池化：

```python
attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
summed = (token_embeddings * attention_mask).sum(dim=1)
counts = attention_mask.sum(dim=1).clamp(min=1e-9)
embeddings = summed / counts
```

这段代码只对非 padding token 求平均，避免 padding 干扰句向量。

归一化：

```python
embeddings = normalize_rows(embeddings)
```

归一化后就可以用点积当余弦相似度。

### 5.4 `HashEmbedder`：本地测试编码器

`HashEmbedder` 用 hash 生成确定性向量：

```python
digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
col = value % self.dim
```

它不理解语义，只是为了在没有下载模型时验证脚本流程。

### 5.5 `load_targets()`：读取 C-Eval 目标集

关键语句：

```python
texts = [str(item.get("target_text", "")).strip() for item in targets]
```

目标集必须有 `target_text`。不带答案和带答案两个 C-Eval JSONL 都满足这个字段。

### 5.6 `build_candidate_text()`：拼接候选样本文本

默认模板：

```yaml
text_template: "指令：{instruction}\n问题：{input}\n回答：{output}"
```

代码：

```python
return template.format(
    instruction=str(record.get("instruction", "")).strip(),
    input=str(record.get("input", "")).strip(),
    output=str(record.get("output", "")).strip(),
).strip()
```

也就是说，候选样本不是只拿问题，而是把指令、问题、回答一起编码。

### 5.7 `flush_batch()`：一批候选样本的相似度计算

候选编码：

```python
candidate_embeddings = embedder.encode(batch_texts)
```

相似度矩阵：

```python
score_matrix = candidate_embeddings @ target_embeddings.T
```

如果 batch size 是 32，C-Eval 目标集是 426 条，那么矩阵形状就是：

```text
32 x 426
```

每条候选取最高分：

```python
best_scores, best_indices = score_matrix.max(dim=1)
```

`best_indices` 用来找到最相似的 C-Eval 题目。

### 5.8 `update_heap()`：维护 Top-K

脚本不用把 50 万条全部排序，而是维护一个最小堆：

```python
if len(heap) < top_k:
    heapq.heappush(heap, item)
    return
if score > heap[0][0]:
    heapq.heapreplace(heap, item)
```

堆顶永远是当前 Top-K 里分数最低的样本。如果新样本比堆顶高，就替换它。

这样内存只需要保存 10 万条入选样本，而不是保存全部 50 万条。

### 5.9 `enrich_record()`：给样本加筛选元信息

输出时每条样本会增加：

```python
enriched["similarity_score"] = round(float(score), 8)
enriched["best_target_id"] = best_target.get("id")
enriched["best_target_subject"] = best_target.get("subject")
enriched["best_target_split"] = best_target.get("split")
enriched["best_target_text"] = best_target.get("target_text")
```

这样后续你可以追踪：这条训练样本为什么被选中，它最像哪道 C-Eval 医学题。

## 6. 输出格式

输出 JSONL 每行保留原始 SFT 字段，并增加筛选元信息：

```json
{
  "instruction": "请回答以下医疗问题",
  "input": "候选样本问题",
  "output": "候选样本回答",
  "similarity_score": 0.8231,
  "best_target_id": "clinical_medicine-test-17",
  "best_target_subject": "clinical_medicine",
  "best_target_split": "test",
  "best_target_text": "科目：临床医学\n题目：..."
}
```

报告文件包含：

- `total_read`：读取候选样本总数。
- `encoded_count`：成功编码样本数。
- `selected_count`：最终保留样本数。
- `score_mean_all`：全部候选平均相似度。
- `score_min_selected`：入选样本最低分。
- `score_max_selected`：入选样本最高分。
- `target_count`：C-Eval 目标题数量。
- `embedding_model`：使用的 embedding 模型。

## 7. 小规模自测

用 hash 后端测试，不需要下载模型：

```bash
python3 scripts/filter_by_ceval_similarity.py \
  --embedding-backend hash \
  --input /private/tmp/medsft_similarity_smoke/alpaca.jsonl \
  --target data/eval/ceval_medical_question_only.jsonl \
  --output /private/tmp/medsft_similarity_smoke/top3.jsonl \
  --report /private/tmp/medsft_similarity_smoke/report.json \
  --top-k 3 \
  --batch-size 2
```

正式语义筛选必须使用：

```bash
--embedding-backend transformers
```

## 8. 实验建议

建议跑两版：

1. 主实验：使用 `ceval_medical_question_only.jsonl`
2. 对照实验：使用 `ceval_medical_question_with_answer.jsonl`

然后比较：

- Top-100k 重合率
- 相似度分布
- `best_target_subject` 中临床医学和基础医学比例
- 抽样人工检查被筛出的样本是否更贴近医学考试知识点
