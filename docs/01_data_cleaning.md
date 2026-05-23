# 01 数据清洗学习文档

## 目标

本阶段要把约 50 万条中文医疗问答语料整理成后续 SFT 可直接使用的数据。清洗脚本采用 JSONL 流式处理方式，一行读入、一行判断、一行写出，避免把全部数据一次性加载进内存。

脚本位置：

```bash
scripts/clean_corpus.py
```

默认配置：

```bash
configs/data_cleaning.yaml
```

默认输入输出：

```text
data/raw/      # 原始 JSONL，不上传 Git
data/cleaned/  # 清洗后 JSONL 和 report，不上传 Git
```

## 支持的数据格式

### 单轮问答

```json
{"question":"高血压患者可以运动吗？","answer":"多数高血压患者可以进行中低强度运动，但应结合血压控制情况和医生建议。"}
```

脚本会自动识别常见字段名，例如 `question` / `answer`、`query` / `response`、`prompt` / `completion`。

### Alpaca

```json
{"instruction":"请回答以下医疗问题","input":"糖尿病患者饮食要注意什么？","output":"建议控制总热量，规律进餐，减少精制糖摄入，并遵医嘱监测血糖。"}
```

清洗后会继续输出为 Alpaca JSONL：

```json
{"instruction":"请回答以下医疗问题","input":"糖尿病患者饮食要注意什么？","output":"建议控制总热量，规律进餐，减少精制糖摄入，并遵医嘱监测血糖。"}
```

### ShareGPT 多轮对话

```json
{"conversations":[{"from":"user","value":"最近总是胸闷怎么办？"},{"from":"assistant","value":"胸闷原因较多，如持续不缓解或伴胸痛、出汗、呼吸困难，应及时就医。"}]}
```

脚本会把角色统一为：

- 用户：`human`
- 助手：`gpt`

并输出为 ShareGPT JSONL。

## 清洗规则

### 1. 字段校验

单轮样本必须能抽取出问题和回答，多轮样本必须能抽取出有效的 `conversations` / `messages`。缺字段、空字段、非 JSON 行会被过滤，并写入 `cleaning_report.json`。

### 2. 文本标准化

脚本会执行：

- HTML 实体反转义
- HTML 标签删除
- Unicode NFKC 规范化
- 控制字符删除
- 零宽字符和 BOM 删除
- 重复空白压缩
- 多余空行压缩

这些规则能减少同一文本因为编码、空白或网页标签不同导致的重复和噪声。

### 3. 广告与联系方式过滤

包含广告关键词或联系方式的样本会被过滤。默认检查：

- URL
- 手机号
- 邮箱
- 微信 / QQ 号
- `加微信`、`免费咨询`、`扫码`、`推广` 等关键词

规则可在 `configs/data_cleaning.yaml` 中调整。

### 4. 长度过滤

默认阈值：

| 字段 | 最小字符数 | 最大字符数 |
| --- | --- | --- |
| 问题 | 4 | 2048 |
| 回答 | 10 | 8192 |

过短样本通常信息量不足，过长样本可能存在拼接错误、网页抓取污染或超出训练长度限制。

### 5. 精确去重

脚本会对标准化后的问题和回答计算 SHA-256 hash。重复样本只保留第一次出现的记录。多轮样本会对全部轮次按角色和值一起计算 hash。

当前阶段只做精确去重。近似去重、embedding 相似度筛选会在后续“领域数据筛选”阶段实现。

### 6. 多轮角色顺序修复

ShareGPT 样本会被修复为：

```text
human -> gpt -> human -> gpt
```

处理规则：

- `user`、`human`、`患者`、`病人` 统一为 `human`。
- `assistant`、`gpt`、`doctor`、`医生`、`客服` 统一为 `gpt`。
- 连续同角色消息会合并。
- 开头不是用户消息时，会删除前置助手消息。
- 结尾不是助手消息时，会删除尾部用户消息。
- 修复后仍不能形成有效交替结构的样本会被过滤。

## 运行方式

将原始 JSONL 文件放入：

```bash
data/raw/
```

运行：

```bash
python scripts/clean_corpus.py \
  --config configs/data_cleaning.yaml \
  --input data/raw \
  --output data/cleaned
```

如果只想清洗单个文件：

```bash
python scripts/clean_corpus.py \
  --config configs/data_cleaning.yaml \
  --input data/raw/medical_qa.jsonl \
  --output data/cleaned
```

## 输出文件

```text
data/cleaned/
├── cleaned_alpaca.jsonl
├── cleaned_sharegpt.jsonl
└── cleaning_report.json
```

字段含义：

- `cleaned_alpaca.jsonl`：单轮 SFT 样本。
- `cleaned_sharegpt.jsonl`：多轮 SFT 样本。
- `cleaning_report.json`：清洗统计报告。

报告中的关键统计：

| 字段 | 含义 |
| --- | --- |
| `total_read` | 总读取样本数 |
| `kept` | 成功保留样本数 |
| `alpaca_kept` | Alpaca 输出样本数 |
| `sharegpt_kept` | ShareGPT 输出样本数 |
| `empty_field` | 空字段过滤数 |
| `too_short` | 过短过滤数 |
| `too_long` | 过长过滤数 |
| `ad_or_contact` | 广告或联系方式过滤数 |
| `duplicate` | 重复过滤数 |
| `invalid_json` | 非法 JSON 行数 |
| `multi_turn_repaired` | 多轮修复次数 |
| `multi_turn_failed` | 多轮修复失败数 |

## 验收标准

本阶段完成后应满足：

- 能流式读取目录下多个 JSONL 文件。
- 能输出 Alpaca 和 ShareGPT 两类清洗结果。
- 能生成 `cleaning_report.json`。
- 重复样本、广告样本、空字段样本、过短过长样本会被过滤。
- 多轮对话能统一角色并修复常见顺序问题。
- `data/raw/` 和 `data/cleaned/` 中的真实数据不会被 Git 提交。

## 常见问题

### 运行时报 `Input path does not exist`

确认输入目录或文件存在。例如：

```bash
mkdir -p data/raw
```

然后把原始 `.jsonl` 文件放进去。

### 输出为空

优先查看 `data/cleaned/cleaning_report.json`。常见原因是字段名不在配置的字段别名里，或长度阈值过严。可以在 `configs/data_cleaning.yaml` 中增加字段名。

### 误过滤太多联系方式

医学问答中可能包含合理的电话样式文本。如果误杀过多，可以先放宽 `contact_patterns`，再通过人工抽样确认风险。

### ShareGPT 样本大量失败

检查原始数据中的角色字段是否是 `from` / `role`，内容字段是否是 `value` / `content`。如果字段不同，需要扩展脚本中的多轮解析逻辑。
