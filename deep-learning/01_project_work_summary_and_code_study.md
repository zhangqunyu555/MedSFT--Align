# MedSFT-Align 已完成工作总结与代码级学习笔记

这份文档用来总结当前项目已经做过的工作，并把关键代码拆开讲清楚。阅读顺序建议是：

1. 先看整体流程，明白这个项目到底在搭一条什么训练链路。
2. 再看每个阶段的输入、输出和真实产物。
3. 最后进入代码级解释，理解每个脚本的关键函数和关键语句。

本项目当前目标是：围绕中文医疗问答场景，参考 `shibing624/MedicalGPT`，基于 `Qwen3-4B-Instruct` 做领域 SFT / QLoRA，并用 C-Eval 医学科目做目标域筛选和评估。

## 1. 现在已经完成了什么

目前已经完成的是“训练前的数据工程 + 训练入口准备 + 评估入口准备”。换句话说，还没有正式开始训练模型，但训练需要的数据、格式、脚本、文档和评估命令已经准备好了。

已经完成的工作包括：

| 阶段 | 当前状态 | 说明 |
| --- | --- | --- |
| 项目初始化 | 已完成 | 已 `git init`，并连接远程仓库 |
| README 与复现计划 | 已完成 | 写了项目目标、路线和预期指标 |
| `.gitignore` | 已完成 | 忽略数据、模型权重、缓存、MedicalGPT 子项目、结果日志 |
| 50 万候选集 | 已完成 | 从 `shibing624/medical` 生成 500000 条 Alpaca 样本 |
| 数据清洗 | 已完成 | 从 500000 条清洗出 381621 条可用 Alpaca 样本 |
| C-Eval 医学目标集 | 已完成 | 构建了临床医学和基础医学的双目标集 |
| 向量相似筛选 | 已完成 | 从 381621 条中筛出 Top 100000 高相似 SFT 样本 |
| Qwen3 训练格式转换 | 已完成 | 转成 MedicalGPT 可读取的 ShareGPT conversations 格式 |
| MedicalGPT 训练说明 | 已完成 | 写了 LoRA / QLoRA SFT 操作说明 |
| lm-evaluation-harness | 已完成准备 | 写了 C-Eval 评估脚本和文档 |
| SwanLab | 已完成准备 | 写了 MedicalGPT + SwanLab 启动脚本 |

## 2. 整体数据流

整个项目的数据流可以理解为一条流水线：

```text
shibing624/medical 原始中文医疗数据
  -> 生成 50 万候选 Alpaca JSONL
  -> 规则清洗，得到 381621 条高质量候选
  -> 下载 C-Eval 临床医学 / 基础医学目标集
  -> 用 embedding 计算候选样本和 C-Eval 目标集的相似度
  -> 选出 similarity_score 最高的 100000 条
  -> 转成 ShareGPT conversations 格式
  -> 交给 MedicalGPT 的 supervised_finetuning.py
  -> 使用 --template_name qwen3 训练 Qwen3-4B-Instruct
  -> 使用 lm-evaluation-harness 在 C-Eval 医学科目上评估
  -> 使用 SwanLab 记录训练曲线和实验配置
```

更具体一点：

```text
shibing624/medical
  -> data/raw/shibing624_medical/medical_zh_500k.jsonl
  -> data/cleaned/shibing624_medical_500k/cleaned_alpaca.jsonl
  -> data/sft/shibing624_medical_top100k.jsonl
  -> MedicalGPT/data/sft_medsft_top100k/train.jsonl
  -> Qwen3-4B-Instruct LoRA / QLoRA SFT
  -> lm-evaluation-harness C-Eval 评估
```

这条链路的核心思想是：不要直接把开源医疗数据全部塞进模型训练，而是先清洗，再按 C-Eval 医学目标域做相似筛选。这样做的目的是让 SFT 数据更贴近临床医学、基础医学这些评测任务，同时减少低质量问答对模型的干扰。

## 3. 当前真实产物与行数

当前已经生成的关键文件如下：

| 文件 | 行数 | 大小 | 说明 |
| --- | ---: | ---: | --- |
| `data/raw/shibing624_medical/medical_zh_500k.jsonl` | 500000 | 346M | 50 万原始候选 Alpaca 样本 |
| `data/cleaned/shibing624_medical_500k/cleaned_alpaca.jsonl` | 381621 | 296M | 清洗后的 Alpaca 样本 |
| `data/sft/shibing624_medical_top100k.jsonl` | 100000 | 109M | 相似度筛选后的 Top 100000 |
| `MedicalGPT/data/sft_medsft_top100k/train.jsonl` | 100000 | 79M | Top 100000 转成 ShareGPT 格式 |
| `MedicalGPT/data/sft_medsft_cleaned_381k/train.jsonl` | 381621 | 311M | 清洗后 381621 条转成 ShareGPT 格式 |

注意：这些真实数据文件都不应该上传 Git。当前 `.gitignore` 已经忽略了：

```text
data/raw/
data/cleaned/
data/sft/
data/eval/
MedicalGPT/
outputs/
results/
swanlog/
```

这很重要，因为训练数据、模型权重、checkpoint、评估结果通常都很大，而且可能包含不适合公开上传的内容。

## 4. 阶段一：项目初始化与文档

### 4.1 这一阶段在做什么

项目一开始不是直接写训练代码，而是先明确复现目标：

- 主模型：`Qwen3-4B-Instruct`
- 数据来源：`shibing624/medical`
- 目标域：C-Eval 的临床医学和基础医学
- 训练方式：LoRA / QLoRA SFT
- 后续路线：DPO / PPO 对齐、C-Eval 评估、PPL 评估、错误案例分析

当前已经写好的基础文档包括：

| 文件 | 作用 |
| --- | --- |
| `README.md` | 项目首页，记录目标、路线、状态 |
| `REPRODUCTION_PLAN.md` | 整体复现思路 |
| `docs/01_data_cleaning.md` | 数据清洗学习文档 |
| `docs/02_ceval_medical_target.md` | C-Eval 目标集学习文档 |
| `docs/03_similarity_filtering.md` | 向量相似筛选学习文档 |
| `docs/04_shibing624_medical_sft_data.md` | 50 万和 10 万数据准备文档 |
| `docs/05_medicalgpt_sft_training.md` | MedicalGPT SFT 训练说明 |
| `docs/06_lm_eval_ceval_swanlab.md` | C-Eval 评估和 SwanLab 文档 |

### 4.2 为什么先写文档

深度学习实验很容易变成“跑过但不可复现”。所以这个项目的做法是每完成一个阶段就写一个文档，记录：

- 输入是什么
- 输出是什么
- 命令怎么跑
- 参数是什么意思
- 产物怎么验收
- 哪些文件不能上传 Git
- 哪些地方容易出错

这样后面换虚拟机、换服务器、换模型时，不需要从聊天记录里翻思路。

## 5. 阶段二：50 万候选集准备

对应脚本：

```text
scripts/prepare_shibing624_medical_sft.py
```

对应配置：

```text
configs/shibing624_medical_500k.yaml
```

### 5.1 这一阶段在做什么

这一阶段从 `shibing624/medical` 的中文 finetune 数据中取出 500000 条有效样本，统一保存成 Alpaca JSONL。

输出文件：

```text
data/raw/shibing624_medical/medical_zh_500k.jsonl
```

每一行大致是：

```json
{
  "instruction": "请回答以下医疗问题",
  "input": "患者的问题或医学问题",
  "output": "医学回答",
  "source": "shibing624/medical",
  "row_id": 123
}
```

这里的 50 万是候选池，不是最终训练集。后面会经过清洗和相似筛选。

### 5.2 关键函数：`open_source()`

职责：打开数据源。

它支持两种来源：

- 本地已经下载好的文件
- Hugging Face 远程 raw URL

为什么要这样设计？因为 `shibing624/medical` 的源文件很大，第一次下载后最好复用本地文件，不要每次都重新拉取。

核心逻辑可以理解为：

```python
if local_path exists:
    open local file
else:
    open remote url
```

这样脚本既能在联网环境下载，也能在已经缓存数据的机器上离线使用。

### 5.3 关键函数：`iter_source_records()`

职责：从源文件中一条条读出样本。

这个函数解决了一个实际问题：源数据可能是两种格式。

第一种是 JSON array：

```json
[
  {"instruction": "...", "input": "...", "output": "..."},
  {"instruction": "...", "input": "...", "output": "..."}
]
```

第二种是 JSONL：

```jsonl
{"instruction": "...", "input": "...", "output": "..."}
{"instruction": "...", "input": "...", "output": "..."}
```

如果一次性 `json.load()` 一个 1GB 级别的大 JSON 文件，会很吃内存。脚本里做了流式读取，避免把完整数据集一次性读进内存。

这个函数背后的思路是：

1. 先看文件第一个非空字符。
2. 如果是 `[`，按 JSON array 流式解析。
3. 如果是 `{` 或普通行文本，按 JSONL 一行一行解析。

### 5.4 关键函数：`normalize_record()`

职责：把不同来源的字段统一成 Alpaca 字段。

它主要检查：

- `instruction` 是否存在
- `input` 是否存在
- `output` 是否存在
- `output` 是否为空

如果样本缺少必要字段，或者答案为空，就跳过。

统一后的字段是：

```text
instruction
input
output
source
row_id
```

这样后续清洗、筛选、训练格式转换都可以只处理一种格式。

### 5.5 关键函数：`prepare_first()`

职责：顺序取前 500000 条有效样本。

这个项目默认采用 `first` 策略，而不是随机抽样。原因是：

- 速度快
- 复现稳定
- 不需要读完整个巨大源文件
- 每次运行结果一致

核心逻辑是：

```python
for record in source:
    normalized = normalize_record(record)
    if normalized is valid:
        write to output
        kept += 1
    if kept >= sample_size:
        break
```

这里的 `sample_size` 是 500000。

### 5.6 关键函数：`prepare_reservoir()`

职责：可选的蓄水池抽样。

蓄水池抽样适合这种场景：

- 总数据量很大
- 不知道总共有多少条
- 又想从全量里随机抽固定数量

它的缺点是必须读完整个源文件，所以当前主流程没有用它。

### 5.7 关键函数：`write_report()`

职责：写报告文件。

报告记录：

- 总读取条数
- 保留条数
- 跳过原因
- 运行时间
- 来源路径
- 输出路径

这类报告很重要，因为后面写论文、项目总结、复现实验时，不能只说“我处理了一些数据”，而要说清楚“读了多少、保留多少、过滤多少、为什么过滤”。

## 6. 阶段三：50 万语料清洗

对应脚本：

```text
scripts/clean_corpus.py
```

对应配置：

```text
configs/data_cleaning.yaml
```

### 6.1 这一阶段在做什么

这一阶段把 500000 条候选样本做规则清洗，最终保留 381621 条。

输入：

```text
data/raw/shibing624_medical/medical_zh_500k.jsonl
```

输出：

```text
data/cleaned/shibing624_medical_500k/cleaned_alpaca.jsonl
```

清洗报告显示，大量样本被过滤的原因包括：

- 问题或回答太短
- 重复
- 包含广告或联系方式
- 问题或回答过长

### 6.2 关键函数：`normalize_text()`

职责：把文本标准化。

它处理的问题包括：

- HTML 实体，例如 `&nbsp;`
- HTML 标签，例如 `<br>`、`<p>`
- 控制字符
- Unicode 全角 / 半角差异
- 多余空格
- 多余空行
- 零宽字符和 BOM

关键语句：

```python
text = html.unescape(text)
text = unicodedata.normalize("NFKC", text)
text = HTML_TAG_RE.sub(" ", text)
text = CONTROL_RE.sub("", text)
```

解释：

- `html.unescape()` 把 HTML 实体转回正常字符。
- `unicodedata.normalize("NFKC", text)` 把一些兼容字符标准化，比如全角字符。
- `HTML_TAG_RE.sub(" ", text)` 删除 HTML 标签。
- `CONTROL_RE.sub("", text)` 删除不可见控制字符。

为什么要做这一步？因为模型训练最怕脏文本。脏文本会让 tokenizer 学到很多无意义模式，还可能导致样本重复判断失败。

### 6.3 关键函数：`contains_ad_or_contact()`

职责：过滤广告和联系方式。

它会检查：

- 广告关键词
- 手机号
- URL
- 邮箱
- 微信 / QQ 联系方式

核心逻辑：

```python
if any(keyword in text for keyword in config["ad_keywords"]):
    return True
return any(re.search(pattern, text) for pattern in config["contact_patterns"])
```

为什么医疗数据里要过滤这些？因为很多医疗问答数据可能来自网页或咨询平台，其中会混入“加微信”“免费咨询”“联系电话”等内容。这些内容不但无助于医学能力，还会让模型学到不安全的引流话术。

### 6.4 关键函数：`validate_lengths()`

职责：过滤过短和过长样本。

关键逻辑：

```python
if q_len == 0 or a_len == 0:
    return "empty_field"
if q_len < min_question_chars or a_len < min_answer_chars:
    return "too_short"
if q_len > max_question_chars or a_len > max_answer_chars:
    return "too_long"
```

为什么要过滤过短样本？

例如：

```text
问题：头疼？
答案：是
```

这种样本信息量太低，容易让模型学到敷衍回答。

为什么要过滤过长样本？

过长样本可能是网页整段复制、说明书、病历堆砌，也可能超过训练时的 `model_max_length`，导致大量截断。

### 6.5 关键函数：`repair_conversations()`

职责：修复多轮对话格式。

虽然当前主训练数据是 Alpaca 单轮格式，但清洗脚本也支持 ShareGPT 多轮数据。

它会处理：

- 角色别名统一，例如 `user`、`患者` -> `human`
- `assistant`、`医生` -> `gpt`
- 连续同角色消息合并
- 开头不是用户消息则裁掉
- 结尾不是助手消息则裁掉

核心思想是：训练数据应该像这样交替出现：

```text
human -> gpt -> human -> gpt
```

如果出现：

```text
gpt -> human -> human -> gpt
```

就需要修复或过滤。

### 6.6 关键函数：`extract_single_turn()`

职责：从一条记录中提取单轮问答。

它支持多种字段：

- QA 格式：`question` / `answer`
- Alpaca 格式：`instruction` / `input` / `output`
- 其他别名：`query`、`prompt`、`response` 等

这样做的好处是：后续如果换一个数据集，不一定需要重写清洗脚本，只要字段名在配置里。

### 6.7 关键函数：`dedup_key_from_pair()`

职责：精确去重。

它会把问题和答案拼起来，再做压缩和哈希。

核心思想：

```python
key = question + answer
key = remove_spaces_and_lowercase(key)
hash(key)
```

为什么不直接用原字符串？因为原字符串可能有空格、换行差异。压缩后再哈希，更容易识别“内容相同但格式略有不同”的重复样本。

### 6.8 关键函数：`clean_corpus()`

职责：主清洗循环。

它是整个清洗脚本的入口，负责：

1. 找到输入 JSONL 文件。
2. 一行一行读取。
3. 判断是单轮还是多轮。
4. 执行文本标准化。
5. 过滤广告、联系方式、过短、过长。
6. 去重。
7. 写入 `cleaned_alpaca.jsonl` 或 `cleaned_sharegpt.jsonl`。
8. 统计清洗报告。

这一阶段最终结果是：

```text
500000 -> 381621
```

也就是说，清洗保留率大约是 76.3%。

## 7. 阶段四：C-Eval 医学双目标集构建

对应脚本：

```text
scripts/build_ceval_medical_jsonl.py
```

### 7.1 这一阶段在做什么

这一阶段下载 C-Eval 中两个医学科目：

- `clinical_medicine`：临床医学
- `basic_medicine`：基础医学

并生成两个目标集：

```text
data/eval/ceval_medical_question_only.jsonl
data/eval/ceval_medical_question_with_answer.jsonl
```

为什么要两个？

因为你明确提出：带答案的是一个数据集，不带答案的是另一个数据集。

不带答案目标集用于：

```text
只根据题干语义做目标域相似筛选
```

带答案目标集用于：

```text
题干 + 正确医学知识点一起做目标域相似筛选
```

当前主实验默认使用不带答案版本，避免答案信息对筛选造成过强引导。

### 7.2 关键函数：`fetch_rows_with_retries()`

职责：分页下载 C-Eval 数据，并在失败时重试。

C-Eval 数据来自 Hugging Face Dataset Viewer API。分页下载比一次性下载更稳：

```text
offset=0, length=100
offset=100, length=100
offset=200, length=100
```

如果网络偶尔失败，重试机制可以避免整个任务直接中断。

### 7.3 关键函数：`iter_dataset_rows()`

职责：遍历所有 subject 和 split。

本项目关注：

```text
clinical_medicine
basic_medicine
```

默认 split 包括：

```text
dev
val
test
```

它会把这些组合全部遍历一遍：

```text
clinical_medicine/dev
clinical_medicine/val
clinical_medicine/test
basic_medicine/dev
basic_medicine/val
basic_medicine/test
```

### 7.4 关键函数：`convert_row()`

职责：把一条 C-Eval 原始记录转换成两个目标记录。

输入通常包含：

```text
question
A
B
C
D
answer
```

输出一：不带答案记录。

```json
{
  "id": "clinical_medicine-val-0",
  "subject": "clinical_medicine",
  "subject_zh": "临床医学",
  "split": "val",
  "question": "原始题干",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "target_text": "科目：临床医学\n题目：原始题干"
}
```

输出二：带答案记录。

```json
{
  "id": "clinical_medicine-val-0",
  "subject": "clinical_medicine",
  "subject_zh": "临床医学",
  "split": "val",
  "question": "原始题干",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "C",
  "answer_text": "正确选项文本",
  "question_with_answer": "原始题干\n正确答案：C. 正确选项文本",
  "target_text": "科目：临床医学\n题目：原始题干\n正确答案：C. 正确选项文本"
}
```

### 7.5 为什么两个目标集用相同 `id`

同一道题在两个文件里使用同一个 `id`，例如：

```text
clinical_medicine-test-17
```

这样后续可以比较：

- 只用题干做 embedding 时，最相似的训练样本是什么
- 题干 + 正确答案做 embedding 时，最相似的训练样本是什么

这对分析筛选策略很有帮助。

### 7.6 风险提醒：C-Eval 不能混入训练标签

C-Eval 可以用于：

- 目标域分布分析
- 相似度筛选参考
- 最终评估

但不能把 C-Eval 的标准答案、解析直接混进 SFT 训练数据。否则评测会被污染，指标就不可信。

## 8. 阶段五：向量相似筛选 Top 100000

对应脚本：

```text
scripts/filter_by_ceval_similarity.py
```

对应配置：

```text
configs/similarity_filtering.yaml
```

### 8.1 这一阶段在做什么

这一阶段把清洗后的 381621 条医疗问答和 C-Eval 医学目标集都转成 embedding 向量，然后计算相似度，选出最接近 C-Eval 医学目标域的 100000 条。

输入：

```text
data/cleaned/shibing624_medical_500k/cleaned_alpaca.jsonl
data/eval/ceval_medical_question_only.jsonl
```

输出：

```text
data/sft/shibing624_medical_top100k.jsonl
```

最终行数：

```text
100000
```

### 8.2 关键类：`TransformersEmbedder`

职责：使用 Hugging Face Transformers 加载 embedding 模型。

默认模型：

```text
BAAI/bge-small-zh-v1.5
```

关键逻辑：

```python
self.tokenizer = AutoTokenizer.from_pretrained(model_name)
self.model = AutoModel.from_pretrained(model_name)
self.model.to(device)
self.model.eval()
```

解释：

- `AutoTokenizer` 负责把文本切成 token。
- `AutoModel` 负责输出 token hidden states。
- `model.eval()` 表示进入推理模式，不更新参数。

### 8.3 embedding 是怎么得到的

在 `encode()` 中，模型输出 `last_hidden_state`：

```python
token_embeddings = output.last_hidden_state
```

它的形状大概是：

```text
batch_size x seq_len x hidden_size
```

脚本用 attention mask 做 mean pooling：

```python
summed = (token_embeddings * attention_mask).sum(dim=1)
counts = attention_mask.sum(dim=1).clamp(min=1e-9)
embeddings = summed / counts
```

意思是：把每个 token 的向量平均起来，得到整段文本的向量。

### 8.4 关键函数：`normalize_rows()`

职责：对向量做 L2 normalize。

```python
return torch.nn.functional.normalize(embeddings, p=2, dim=1)
```

做完 L2 normalize 后，每个向量长度都是 1。此时两个向量的点积就等价于余弦相似度。

也就是说：

```python
cosine_similarity = a @ b
```

### 8.5 关键类：`HashEmbedder`

职责：不加载真实 embedding 模型，用哈希向量做 smoke test。

这个类不是为了正式实验，而是为了快速测试脚本流程：

- 不需要下载模型
- 不需要 GPU
- 不需要 transformers
- 可以快速验证输入输出和 Top-K 逻辑

正式筛选时仍然应该使用 `TransformersEmbedder`。

### 8.6 关键函数：`load_targets()`

职责：读取 C-Eval 目标集。

它要求每条目标记录都有：

```text
target_text
```

因为 embedding 用的是 `target_text`，不是原始 `question`。

不带答案目标集的 `target_text`：

```text
科目：临床医学
题目：原始题干
```

带答案目标集的 `target_text`：

```text
科目：临床医学
题目：原始题干
正确答案：C. 正确选项文本
```

### 8.7 关键函数：`build_candidate_text()`

职责：把一条 Alpaca 候选样本拼成用于 embedding 的文本。

默认模板：

```text
指令：{instruction}
问题：{input}
回答：{output}
```

为什么把回答也放进去？

因为我们筛选的是 SFT 样本，不只是问题。一个样本是否适合医学 SFT，不仅取决于问题像不像医学题，也取决于答案是否包含相关医学知识。

### 8.8 核心语句：相似度矩阵

最核心的计算是：

```python
scores = candidate_embeddings @ target_embeddings.T
```

假设：

- candidate batch 有 32 条
- C-Eval 目标集有 426 条

那么矩阵形状是：

```text
32 x 426
```

每一行代表一个候选样本和所有 C-Eval 题目的相似度。

然后取最大值：

```python
best_scores, best_indices = scores.max(dim=1)
```

意思是：每条候选样本只保留它最像的那道 C-Eval 题，以及对应分数。

### 8.9 关键函数：`update_heap()`

职责：用最小堆维护 Top-K。

为什么用堆？

因为候选样本有 381621 条，如果每条都存下来最后排序也可以，但更通用的做法是维护一个大小为 K 的最小堆：

- 堆还没满：直接加入
- 堆满了：如果新样本分数高于堆顶，就替换堆顶
- 堆顶永远是当前 Top-K 里分数最低的那条

这样适合更大的数据规模，例如 500 万、5000 万。

### 8.10 关键函数：`enrich_record()`

职责：给入选样本增加筛选元信息。

输出样本会保留原始 Alpaca 字段，并新增：

```text
similarity_score
best_target_id
best_target_subject
best_target_split
best_target_text
```

这样后面可以追踪：

```text
为什么这条训练样本被选中？
它最像哪一道 C-Eval 医学题？
```

这对错误分析和实验解释非常重要。

## 9. 阶段六：转换成 Qwen3 / MedicalGPT 训练格式

对应脚本：

```text
scripts/convert_alpaca_to_sharegpt.py
```

### 9.1 这一阶段在做什么

当前数据经过清洗和筛选后仍然是 Alpaca JSONL：

```json
{
  "instruction": "请回答以下医疗问题",
  "input": "问题",
  "output": "答案"
}
```

但是 `MedicalGPT/training/supervised_finetuning.py` 的预处理逻辑主要读取：

```text
conversations
```

所以训练前要转成 ShareGPT 格式：

```json
{
  "conversations": [
    {"from": "human", "value": "问题"},
    {"from": "gpt", "value": "答案"}
  ]
}
```

注意：这不是手写 Qwen3 特殊 token。真正的 Qwen3 模板由 MedicalGPT 的：

```bash
--template_name qwen3
```

在训练时处理。

### 9.2 为什么不用 `MedicalGPT/tools/convert_dataset.py`

一开始尝试使用：

```bash
python MedicalGPT/tools/convert_dataset.py ...
```

但是当前环境缺少：

```text
datasets
```

报错：

```text
ModuleNotFoundError: No module named 'datasets'
```

为了不被依赖卡住，我们新增了一个无第三方依赖的流式转换脚本：

```text
scripts/convert_alpaca_to_sharegpt.py
```

### 9.3 关键函数：`build_user_prompt()`

职责：把 `instruction` 和 `input` 拼成用户输入。

关键逻辑：

```python
if instruction and input_text:
    return f"{instruction}\n\n{input_text}"
return instruction or input_text
```

解释：

- 如果两个字段都有，就中间加两个换行拼接。
- 如果只有一个字段，就用那个字段。

这样兼容两种 Alpaca 样本：

```text
instruction 有内容，input 为空
instruction 是任务描述，input 是具体问题
```

### 9.4 关键函数：`convert_record()`

职责：把一条 Alpaca 记录转成 ShareGPT。

核心逻辑：

```python
instruction = clean_text(record.get("instruction"))
input_text = clean_text(record.get("input"))
output = clean_text(record.get("output"))
user_prompt = build_user_prompt(instruction, input_text)

if not user_prompt or not output:
    return None

return {
    "conversations": [
        {"from": "human", "value": user_prompt},
        {"from": "gpt", "value": output},
    ]
}
```

这里最重要的是：

- 用户问题必须非空
- 助手答案必须非空
- 输出字段必须叫 `conversations`
- 用户角色用 `human`
- 助手角色用 `gpt`

这正好对齐 MedicalGPT 的训练预处理逻辑。

### 9.5 关键函数：`convert_file()`

职责：流式转换整个 JSONL 文件。

它一行一行读，不一次性加载全部数据：

```python
for line in fin:
    record = json.loads(line)
    converted = convert_record(record)
    fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
```

同时它会统计：

```text
total_read
kept
skipped
bad_json
elapsed_seconds
```

并且每隔 `log_every` 行打印进度：

```text
[progress] read=10000 kept=10000 skipped=0 speed=159752.6 rows/s
```

这个进度日志是现在项目的约定：凡是长时间处理脚本，都应该有进度输出，避免用户不知道程序是不是卡住了。

### 9.6 当前转换结果

Top 100000：

```text
MedicalGPT/data/sft_medsft_top100k/train.jsonl
100000 valid
```

清洗后 381621：

```text
MedicalGPT/data/sft_medsft_cleaned_381k/train.jsonl
381621 valid
```

两份都已经逐行 JSON 校验通过。

## 10. 阶段七：MedicalGPT / Qwen3 SFT 训练准备

对应文档：

```text
docs/05_medicalgpt_sft_training.md
```

对应外部子项目：

```text
MedicalGPT/
```

### 10.1 这一阶段在做什么

这一阶段没有正式训练，而是把训练路线准备好：

- 使用 `MedicalGPT/training/supervised_finetuning.py`
- 模型使用 `Qwen/Qwen3-4B-Instruct`
- 数据使用刚转换好的 ShareGPT JSONL
- 微调方式使用 LoRA / QLoRA
- 模板使用 `--template_name qwen3`

### 10.2 为什么 `MedicalGPT/` 加入 `.gitignore`

`MedicalGPT/` 是外部参考项目，不属于当前根仓库源码的一部分。它里面还有自己的 `.git`、脚本、数据和输出目录。

如果不忽略它，根仓库会试图把整个外部项目提交进去，既混乱又容易把大文件带上。

所以 `.gitignore` 中加入了：

```text
MedicalGPT/
medical-gpt/
```

### 10.3 Qwen3 格式到底在哪里处理

训练数据文件本身是 ShareGPT：

```json
{
  "conversations": [
    {"from": "human", "value": "问题"},
    {"from": "gpt", "value": "答案"}
  ]
}
```

Qwen3 的特殊对话模板在训练命令中指定：

```bash
--template_name qwen3
```

也就是说，不需要我们手写：

```text
<|im_start|>user
...
<|im_end|>
```

MedicalGPT 会在预处理阶段把 `conversations` 转成模型需要的 prompt。

### 10.4 主实验和对照实验

主实验推荐：

```bash
--train_file_dir data/sft_medsft_top100k
```

因为这 10 万条经过了 C-Eval 相似筛选。

对照实验：

```bash
--train_file_dir data/sft_medsft_cleaned_381k
```

这样可以比较：

```text
相似筛选 10 万 vs 清洗后全部 38 万
```

哪个对 C-Eval 医学准确率更有帮助。

## 11. 阶段八：lm-evaluation-harness C-Eval 评估准备

对应脚本：

```text
scripts/run_ceval_lm_eval.sh
```

对应文档：

```text
docs/06_lm_eval_ceval_swanlab.md
```

### 11.1 这一阶段在做什么

这一阶段准备使用 `lm-evaluation-harness` 来评估模型在 C-Eval 医学科目上的表现。

评估对象包括：

- 原始 `Qwen/Qwen3-4B-Instruct`
- SFT 后的 LoRA / QLoRA adapter
- 可选的合并后完整模型

主评估科目：

```text
clinical_medicine
basic_medicine
```

### 11.2 关键参数：`--model`

脚本中默认：

```bash
MODEL="Qwen/Qwen3-4B-Instruct"
```

这表示评估 Hugging Face 上的原始 Qwen3-4B-Instruct。

也可以改成本地路径：

```bash
--model /path/to/local/Qwen3-4B-Instruct
```

### 11.3 关键参数：`--adapter`

如果要评估 LoRA / QLoRA adapter，就传：

```bash
--adapter MedicalGPT/outputs/qwen3_4b_medical_qlora_top100k
```

脚本会把它拼进：

```text
peft=adapter_path
```

最终 model args 形如：

```text
pretrained=Qwen/Qwen3-4B-Instruct,trust_remote_code=True,dtype=bfloat16,peft=MedicalGPT/outputs/...
```

这表示 harness 会加载 base model，再加载 PEFT adapter。

### 11.4 关键参数：`--tasks`

默认：

```bash
ceval-valid_clinical_medicine,ceval-valid_basic_medicine
```

但不同版本的 harness 任务名可能不同，所以文档里要求先查：

```bash
lm-eval ls tasks | grep -Ei "ceval|clinical|basic"
```

如果实际任务名是旧版风格，就用 `--tasks` 覆盖。

### 11.5 关键参数：`--limit 5`

`--limit 5` 是 smoke test。

它的意义是：

- 验证模型能加载
- 验证任务名正确
- 验证数据能下载
- 验证输出目录能写
- 避免一上来完整评估浪费时间

正式评估时去掉 `--limit`。

## 12. 阶段九：SwanLab 训练日志准备

对应脚本：

```text
scripts/run_medicalgpt_sft_swanlab.sh
```

### 12.1 这一阶段在做什么

这一阶段是让 MedicalGPT 的训练过程能在 SwanLab 里可视化。

检查结果显示：`MedicalGPT/` 本身没有 SwanLab 专用代码，默认是：

```bash
--report_to tensorboard
```

但是因为 MedicalGPT 使用 Hugging Face `Trainer`，所以只要 transformers 版本支持 SwanLab，就可以直接改成：

```bash
--report_to swanlab
```

### 12.2 为什么不改 MedicalGPT 源码

不改源码的好处：

- 外部子项目保持干净
- 后续更新 MedicalGPT 更容易
- 根项目只维护自己的启动脚本
- 不会破坏原项目训练逻辑

所以我们在根项目写了：

```text
scripts/run_medicalgpt_sft_swanlab.sh
```

这个脚本进入 `MedicalGPT/` 后调用：

```bash
python training/supervised_finetuning.py
```

### 12.3 关键参数：`--report_to swanlab`

这是 SwanLab 集成的核心。

它会让 Hugging Face `Trainer` 把训练日志上报给 SwanLab。

### 12.4 关键参数：`--run_name`

用于区分不同实验：

```bash
--run_name qwen3-4b-medical-qlora-top100k
```

如果你跑对照实验，可以改成：

```bash
RUN_NAME=qwen3-4b-medical-cleaned381k
```

### 12.5 SwanLab 可以记录什么

通过 `Trainer` 自动记录的内容通常包括：

| 类型 | 指标 |
| --- | --- |
| 训练曲线 | `loss`、`learning_rate`、`epoch`、`step` |
| 验证指标 | `eval_loss`、`eval_runtime`、`eval_samples_per_second` |
| PPL | 如果训练脚本计算 `perplexity`，也会记录 |
| 性能指标 | `train_runtime`、`train_samples_per_second` |
| 配置参数 | batch size、学习率、epoch、output_dir、save_steps |
| 运行信息 | run name、项目名、环境信息 |

在 MedicalGPT 的 `supervised_finetuning.py` 里，训练后会调用：

```python
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
```

评估后会调用：

```python
metrics = trainer.evaluate(metric_key_prefix="eval")
metrics["perplexity"] = perplexity
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
```

所以 SwanLab 能看到这些 trainer 上报的指标。

### 12.6 小样本验证

脚本支持通过环境变量改参数：

```bash
MAX_TRAIN_SAMPLES=200 \
MAX_EVAL_SAMPLES=20 \
OUTPUT_DIR=outputs/qwen3_4b_medical_swanlab_smoke \
RUN_NAME=qwen3-4b-medical-swanlab-smoke \
bash scripts/run_medicalgpt_sft_swanlab.sh
```

这比直接跑完整 10 万更稳。

## 13. 常见坑总结

### 13.1 JSONL 不能用 `python -m json.tool` 整文件校验

JSONL 是一行一个 JSON 对象，不是一个完整 JSON 数组。

所以这样会报错：

```bash
python -m json.tool file.jsonl
```

常见错误：

```text
Extra data: line 2 column 1
```

这是正常的。正确做法是逐行 `json.loads()`。

### 13.2 Qwen3 训练格式不是手写特殊 token

不要把训练文件手动写成：

```text
<|im_start|>user ...
```

当前正确流程是：

```text
Alpaca
  -> ShareGPT conversations
  -> MedicalGPT supervised_finetuning.py
  -> --template_name qwen3
  -> 内部转 Qwen3 prompt
```

### 13.3 C-Eval 任务名可能随 harness 版本变化

所以先查：

```bash
lm-eval ls tasks | grep -Ei "ceval|clinical|basic"
```

再跑正式评估。

### 13.4 大文件和外部项目不能上传 Git

当前必须忽略：

```text
data/
MedicalGPT/
outputs/
results/
swanlog/
```

如果不忽略，仓库会变得非常大，而且可能包含不该公开的数据和模型文件。

### 13.5 长时间脚本必须有进度日志

当前已经在以下脚本里体现：

- `filter_by_ceval_similarity.py`：有 `log_every`
- `convert_alpaca_to_sharegpt.py`：每 N 行打印 read / kept / skipped / speed

后续训练、评估、数据生成脚本也应该保留这种习惯。

## 14. 当前可以怎么开始训练

主实验训练数据：

```text
MedicalGPT/data/sft_medsft_top100k/train.jsonl
```

进入 MedicalGPT：

```bash
cd MedicalGPT
```

启动 Qwen3 SFT 时关键参数是：

```bash
--model_name_or_path Qwen/Qwen3-4B-Instruct
--train_file_dir data/sft_medsft_top100k
--validation_file_dir data/sft_medsft_top100k
--template_name qwen3
--use_peft True
```

如果用 SwanLab 脚本，从根目录运行：

```bash
bash scripts/run_medicalgpt_sft_swanlab.sh
```

如果想先小样本验证：

```bash
MAX_TRAIN_SAMPLES=200 \
MAX_EVAL_SAMPLES=20 \
OUTPUT_DIR=outputs/qwen3_4b_medical_swanlab_smoke \
RUN_NAME=qwen3-4b-medical-swanlab-smoke \
bash scripts/run_medicalgpt_sft_swanlab.sh
```

## 15. 当前可以怎么评估

先评估原始 Qwen3：

```bash
bash scripts/run_ceval_lm_eval.sh --limit 5
```

正式评估：

```bash
bash scripts/run_ceval_lm_eval.sh
```

评估 SFT adapter：

```bash
bash scripts/run_ceval_lm_eval.sh \
  --adapter MedicalGPT/outputs/qwen3_4b_medical_qlora_top100k \
  --output results/ceval/qwen3_4b_medical_qlora_top100k
```

最终希望比较：

```text
原始 Qwen3-4B-Instruct C-Eval 医学准确率
vs
10 万高相似 SFT 后 C-Eval 医学准确率
```

项目预期目标是：

```text
0.8324 -> 0.8652
```

## 16. 你现在已经具备的复现资产

到目前为止，你已经不只是“有一份数据”，而是有了一整套可复现资产：

- 可复现的数据下载脚本
- 可复现的数据清洗脚本
- 可复现的 C-Eval 目标集构建脚本
- 可复现的 embedding 相似筛选脚本
- 可复现的 Alpaca 到 ShareGPT 转换脚本
- 可复现的 MedicalGPT 训练说明
- 可复现的 C-Eval 评估脚本
- 可复现的 SwanLab 训练启动脚本
- 每个阶段对应的学习文档

这意味着后续真正训练模型时，如果结果不好，可以回头定位是哪一层的问题：

- 数据太脏？
- 相似筛选策略不合适？
- Qwen3 模板不对？
- LoRA 参数太弱？
- 学习率太大？
- C-Eval 任务名或评估方式不一致？

这就是这个项目目前最重要的工程价值：它不是一次性脚本，而是一条能反复实验和对照的后训练流水线。

