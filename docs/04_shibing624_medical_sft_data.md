# 04 shibing624/medical 50 万与 10 万 SFT 数据

## 1. 本阶段目标

本阶段要落地两个明确大小的数据集：

- 50 万候选集：`data/raw/shibing624_medical/medical_zh_500k.jsonl`
- 10 万筛选集：`data/sft/shibing624_medical_top100k.jsonl`

这里的 `50w` 和 `10w` 都是数据条数：

```text
50w = 500000 行
10w = 100000 行
```

10 万筛选集必须从这 50 万候选集中筛出来，不能直接从 `shibing624/medical` 全量数据里筛。

## 2. 数据来源

使用 Hugging Face 数据集：

```text
shibing624/medical
```

中文 SFT 数据文件：

```text
finetune/train_zh_0.json
```

脚本默认下载地址：

```text
https://huggingface.co/datasets/shibing624/medical/resolve/main/finetune/train_zh_0.json
```

这个源文件很大，正式运行时需要联网和足够磁盘空间。脚本支持边读边写，不会把全量数据一次性加载进内存。

## 3. 三步流程

### 3.1 生成 50 万候选集

```bash
python3 scripts/prepare_shibing624_medical_sft.py \
  --config configs/shibing624_medical_500k.yaml \
  --sample-size 500000 \
  --output data/raw/shibing624_medical/medical_zh_500k.jsonl
```

输出：

```text
data/raw/shibing624_medical/medical_zh_500k.jsonl
data/raw/shibing624_medical/medical_zh_500k_report.json
```

### 3.2 清洗 50 万候选集

```bash
python3 scripts/clean_corpus.py \
  --input data/raw/shibing624_medical/medical_zh_500k.jsonl \
  --output data/cleaned/shibing624_medical_500k
```

输出：

```text
data/cleaned/shibing624_medical_500k/cleaned_alpaca.jsonl
data/cleaned/shibing624_medical_500k/cleaning_report.json
```

### 3.3 从 50 万中筛出 10 万

```bash
python3 scripts/filter_by_ceval_similarity.py \
  --config configs/similarity_filtering.yaml \
  --input data/cleaned/shibing624_medical_500k/cleaned_alpaca.jsonl \
  --target data/eval/ceval_medical_question_only.jsonl \
  --output data/sft/shibing624_medical_top100k.jsonl \
  --report data/sft/shibing624_medical_top100k_report.json \
  --top-k 100000
```

输出：

```text
data/sft/shibing624_medical_top100k.jsonl
data/sft/shibing624_medical_top100k_report.json
```

## 4. 50 万候选集格式

每行是一个 Alpaca 样本：

```json
{
  "instruction": "请回答以下医疗问题",
  "input": "用户问题",
  "output": "医生或助手回答",
  "source": "shibing624/medical",
  "row_id": 12345
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `instruction` | 指令 |
| `input` | 用户问题 |
| `output` | 医疗回答 |
| `source` | 数据来源 |
| `row_id` | 源文件中的行号或样本顺序 |

## 5. 抽样策略

配置文件：

```bash
configs/shibing624_medical_500k.yaml
```

默认：

```yaml
sample_size: 500000
strategy: first
seed: 42
```

### 5.1 `first`

从源数据中顺序读取前 500000 条有效样本。优点是快、稳定、复现简单。

### 5.2 `reservoir`

从全量源数据中做蓄水池抽样。优点是分布更随机；缺点是必须读完整个大文件，耗时更长。

## 6. 关键函数学习

### 6.1 `open_source()`

功能：打开本地源文件或远程 Hugging Face URL。

如果配置了 `local_source_path`，优先读本地文件；否则从 `source_url` 读取。

### 6.2 `iter_source_records()`

功能：自动识别源数据是 JSON array 还是 JSONL。

- 如果第一个非空白字符是 `[`，按 JSON array 流式解析。
- 否则按 JSONL 逐行解析。

这样脚本可以兼容不同下载格式。

### 6.3 `normalize_record()`

功能：把源数据转换成项目需要的 Alpaca 字段。

关键规则：

- `output` 不能为空。
- `instruction` 为空时补成 `请回答以下医疗问题`。
- 保留 `source` 和 `row_id` 方便追溯。

### 6.4 `prepare_first()`

功能：生成前 500000 条有效样本。

核心逻辑：

```python
if stats["kept"] >= sample_size:
    break
```

写够 500000 条就停止，不继续下载或解析后面的数据。

### 6.5 `prepare_reservoir()`

功能：做固定随机种子的蓄水池抽样。

适合你希望 50 万样本更随机时使用。

## 7. 小规模验证

先跑 20 条测试：

```bash
python3 scripts/prepare_shibing624_medical_sft.py \
  --sample-size 20 \
  --output /private/tmp/shibing624_medical_20.jsonl \
  --report /private/tmp/shibing624_medical_20_report.json
```

检查行数：

```bash
wc -l /private/tmp/shibing624_medical_20.jsonl
```

预期输出：

```text
20 /private/tmp/shibing624_medical_20.jsonl
```

## 8. 正式验收

50 万验收：

```bash
wc -l data/raw/shibing624_medical/medical_zh_500k.jsonl
```

必须是：

```text
500000 data/raw/shibing624_medical/medical_zh_500k.jsonl
```

10 万验收：

```bash
wc -l data/sft/shibing624_medical_top100k.jsonl
```

必须是：

```text
100000 data/sft/shibing624_medical_top100k.jsonl
```

## 9. 常见问题

### 9.1 下载很慢

`train_zh_0.json` 文件约 GB 级，网络慢是正常的。可以先手动下载到本地，然后运行：

```bash
python3 scripts/prepare_shibing624_medical_sft.py \
  --local-source-path /path/to/train_zh_0.json \
  --sample-size 500000
```

### 9.2 行数不足 500000

脚本会直接报错。原因通常是源文件不是预期的中文 SFT 文件，或有效样本不足。

### 9.3 10 万筛选太慢

正式筛选会加载 embedding 模型。可以先用 hash 后端验证流程：

```bash
python3 scripts/filter_by_ceval_similarity.py \
  --embedding-backend hash \
  --input data/cleaned/shibing624_medical_500k/cleaned_alpaca.jsonl \
  --target data/eval/ceval_medical_question_only.jsonl \
  --output /private/tmp/shibing624_top100.jsonl \
  --report /private/tmp/shibing624_top100_report.json \
  --top-k 100
```
