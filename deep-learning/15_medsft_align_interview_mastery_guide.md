# MedSFT-Align 面试必会与源码掌握手册

这份手册不是简单复述项目流程，而是回答三个更实际的问题：

1. 面试时哪些内容只需要能讲清楚？
2. 哪些内容必须掌握到源码和关键语句？
3. 哪些能力是自己实现的，哪些是基于开源框架完成的？

项目中不需要背下每一行代码，但简历上作为项目亮点出现的模块，至少要做到：

- 能画出输入、处理、输出的数据流。
- 能解释核心算法和关键公式。
- 能定位真实源码中的关键函数。
- 能解释重要变量、张量和数据结构。
- 能回答为什么这样设计，以及它有什么局限。
- 能说出一次实际遇到的错误和解决方法。

---

## 一、项目全链路与个人贡献边界

### 1.1 项目全链路

MedSFT-Align 的完整流程是：

```text
shibing624/medical 中文医疗数据
  -> 抽取 500000 条候选 Alpaca 数据
  -> 字段校验、规范化、去重、广告过滤、长度过滤
  -> 得到 381621 条清洗数据
  -> 构建 C-Eval 临床医学、基础医学目标集
  -> 使用 BGE 对候选集和目标集编码
  -> 计算每条候选与目标集的最大余弦相似度
  -> 最小堆保留 Top 100000
  -> 转成 ShareGPT conversations
  -> Qwen3-4B-Instruct-2507 QLoRA SFT
  -> C-Eval 与 answer-only PPL 评测
  -> 从 Top100k 派生 5000 条复杂病例 PPO 数据
  -> 格式分 + 准确率分 + 安全分
  -> TRL PPO 强化对齐
  -> C-Eval、PPL、格式指令遵循、安全覆盖率评测
```

真实数据报告是：

| 阶段 | 数量或结果 |
| --- | ---: |
| 原始候选 | 500000 |
| 清洗保留 | 381621 |
| 过短过滤 | 115194 |
| 重复过滤 | 2783 |
| 广告或联系方式过滤 | 361 |
| 过长过滤 | 41 |
| C-Eval 医学目标 | 426 |
| 相似筛选输出 | 100000 |
| Top-K 最低入选分数 | 0.619218 |
| PPO 复杂病例数据 | 5000 |

### 1.2 个人贡献边界

面试时建议主动分成三类，不要把调用框架说成从零实现。

| 类型 | 项目内容 | 推荐表述 |
| --- | --- | --- |
| 亲手实现 | 50 万数据准备、清洗规则、格式统一、C-Eval 双目标集、embedding 筛选、Top-K、PPO 数据构造、多维规则奖励、PPL 和格式评测 | “我根据医疗任务需求设计并实现了这些脚本。” |
| 基于开源代码改造 | MedicalGPT SFT 数据适配、Qwen3 template、QLoRA 配置、TRL PPO 的 reward/value wrapper | “我基于 MedicalGPT、Transformers、PEFT 和 TRL 完成适配与扩展。” |
| 直接使用框架 | Trainer 反向传播、AdamW、学习率调度、LoRA 层注入底层实现、PPO 优化器内部更新 | “底层训练循环由开源框架提供，我掌握调用链、损失和关键接口。” |

### 1.3 30 秒项目回答

> 我围绕 Qwen3-4B-Instruct 做了中文医疗问答后训练。先从 shibing624/medical 准备 50 万条候选数据，通过字段校验、文本规范化、广告过滤、长度过滤和精确去重得到 381621 条。为了让训练分布贴近医学评测，我用 BGE 对清洗数据和 C-Eval 临床医学、基础医学题目编码，通过余弦相似度和最小堆筛出 Top 10 万。之后基于 MedicalGPT、Transformers 和 PEFT 做 QLoRA SFT，再从 Top100k 构造 5K 复杂病例，用格式、准确率和安全三维规则奖励做 PPO。评测包括 C-Eval、answer-only PPL 和复杂病例格式指令遵循。

### 1.4 不能错误声称的内容

- 不能说自己从零实现了 Transformers Trainer。
- 不能说自己从零实现了 PEFT 的 LoRA 注入。
- 不能说训练了神经网络 reward model。主实验使用的是手写规则奖励。
- 不能说 DPO、GRPO 是项目主实验，它们只是学习和对照方向。
- 不能把 C-Eval 题目答案作为 SFT 标签混入训练集。
- 不能把带格式提示下的准确率说成“无提示自发格式准确率”。
- `72% -> 94%` 是简历采用的实验口径，需要能提供对应评测协议或人工抽检证据；当前本地自动调试结果不能与它混为一谈。

---

## 二、数据处理必须掌握的原理、源码和答案

## 2.1 为什么采用 JSONL 流式处理

JSONL 每行是一个独立 JSON 对象。它适合 50 万规模数据，因为不需要一次把整个文件读进内存。

清洗主循环位于 `scripts/clean_corpus.py`：

```python
with file_path.open("r", encoding="utf-8") as input_file:
    for line_number, line in enumerate(input_file, start=1):
        if not line.strip():
            continue
        report["total_read"] += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            report["invalid_json"] += 1
            continue
        if not isinstance(record, dict):
            report["invalid_record"] += 1
            continue
```

逐段解释：

- `for line_number, line in enumerate(...)`：一次只读取一行，内存复杂度接近单条样本大小。
- `json.loads(line)`：单行解析失败只过滤当前样本，不让整个任务中断。
- `Counter report`：记录每一种过滤原因，保证清洗过程可审计。
- `line_number`：出现坏数据时可以定位到源文件具体行。

面试标准回答：

> 50 万条数据虽然可以勉强整体加载，但流式 JSONL 更稳定。它把内存占用从 O(N) 降到接近 O(1)，单条 JSON 损坏也不会影响其他样本，并且适合边读、边清洗、边写结果。唯一需要常驻内存的是精确去重集合。

可能追问：去重集合是不是仍然会占内存？

> 是。当前实现把 SHA256 key 放在 `set` 中，因此去重部分是 O(N) 内存。50 万规模可以接受；如果扩展到亿级，可以使用 Bloom Filter、磁盘 KV、分桶排序或数据库唯一索引。

## 2.2 文本规范化

真实函数：

```python
def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = SPACE_RE.sub(" ", text)
    text = "\n".join(part.strip() for part in text.splitlines())
    text = BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()
```

代码解释：

- `html.unescape()`：把 `&nbsp;`、`&lt;` 等 HTML 实体还原。
- `unicodedata.normalize("NFKC", text)`：统一全角、兼容字符等 Unicode 表示。
- `HTML_TAG_RE.sub()`：删除网页标签。
- `CONTROL_RE.sub()`：移除不可见控制字符。
- `\u200b` 是零宽空格，`\ufeff` 常见于 BOM。
- 连续横向空白压成一个空格，连续三个以上换行压成两个。

为什么使用 NFKC：

> 中文医疗数据来自不同网站，数字、括号和英文字母可能存在全角与半角差异。NFKC 可以减少视觉相同但编码不同的问题，有利于去重和 tokenizer 输入一致性。但如果任务依赖特殊排版符号，就要谨慎使用。

## 2.3 字段识别与格式兼容

配置支持多种别名：

```yaml
question_fields: [question, query, prompt, ask, title]
answer_fields: [answer, response, output, completion, reply]
instruction_fields: [instruction]
input_fields: [input]
output_fields: [output]
conversation_fields: [conversations, messages]
```

单轮字段提取代码：

```python
def extract_single_turn(record: dict[str, Any], config: dict[str, Any]) -> tuple[str, str, str]:
    instruction = normalize_text(pick_first(record, config["instruction_fields"]))
    input_text = normalize_text(pick_first(record, config["input_fields"]))
    output = normalize_text(pick_first(record, config["output_fields"]))

    if instruction and output:
        question = normalize_text(f"{instruction}\n{input_text}" if input_text else instruction)
        return instruction, input_text, output

    question = normalize_text(pick_first(record, config["question_fields"]))
    answer = normalize_text(pick_first(record, config["answer_fields"]))
    return "请回答以下医疗问题", question, answer
```

这里完成两种输入的统一：

- 已经是 Alpaca 的数据直接保留 `instruction/input/output`。
- 普通 `question/answer` 数据转换成默认 instruction 加问题和答案。

最终 Alpaca 格式：

```json
{
  "instruction": "请回答以下医疗问题",
  "input": "患者出现胸痛和呼吸困难应该怎么办？",
  "output": "需要警惕急性心血管事件，应尽快就医评估……"
}
```

## 2.4 长度过滤

真实函数：

```python
def validate_lengths(question: str, answer: str, config: dict[str, Any]) -> str | None:
    q_len = count_content_chars(question)
    a_len = count_content_chars(answer)
    if q_len == 0 or a_len == 0:
        return "empty_field"
    if q_len < config["min_question_chars"] or a_len < config["min_answer_chars"]:
        return "too_short"
    if q_len > config["max_question_chars"] or a_len > config["max_answer_chars"]:
        return "too_long"
    return None
```

项目配置：

```text
问题：4 到 2048 个非空白字符
答案：10 到 8192 个非空白字符
```

为什么过滤过短样本：

- 极短答案常是“是”“不是”“好的”，医学信息密度低。
- 可能是网页残片或解析失败。
- 会让模型学习敷衍回答。

为什么过滤过长样本：

- 可能是整页网页、论文或多条记录错误拼接。
- 训练时会被截断，造成答案尾部缺失。
- 长样本显著增加 token 成本并挤占 batch。

局限：

> 字符长度只是启发式质量规则，不等于 token 长度和医学质量。更严谨的方案应增加 token 长度、语言检测、困惑度、分类器和人工抽检。

## 2.5 广告与联系方式过滤

真实函数：

```python
def contains_ad_or_contact(text: str, config: dict[str, Any]) -> bool:
    if any(keyword and keyword in text for keyword in config["ad_keywords"]):
        return True
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in config["contact_patterns"]
    )
```

项目同时使用：

- 关键词：`加微信`、`免费咨询`、`推广` 等。
- 正则：URL、手机号、邮箱、微信号、QQ。

面试追问：误杀怎么办？

> 规则清洗追求精度和召回率平衡。电话号码可能出现在医学示例中，简单正则会误杀。改进方式是记录过滤样本做抽检，对高误杀规则增加上下文条件，或者只脱敏联系方式而不是整条删除。

## 2.6 多轮角色顺序修复

真实函数核心：

```python
def repair_conversations(
    conversations: Any, config: dict[str, Any]
) -> tuple[list[dict[str, str]] | None, bool]:
    if not isinstance(conversations, list):
        return None, False

    repaired: list[dict[str, str]] = []
    changed = False
    for message in conversations:
        if not isinstance(message, dict):
            changed = True
            continue
        role = normalize_role(message_role(message), config)
        value = normalize_text(message_value(message))
        if role is None or not value:
            changed = True
            continue
        if repaired and repaired[-1]["from"] == role:
            repaired[-1]["value"] = normalize_text(
                repaired[-1]["value"] + "\n" + value
            )
            changed = True
        else:
            repaired.append({"from": role, "value": value})

    while repaired and repaired[0]["from"] != "human":
        repaired.pop(0)
        changed = True
    while repaired and repaired[-1]["from"] != "gpt":
        repaired.pop()
        changed = True

    if len(repaired) < 2:
        return None, changed
    for index, message in enumerate(repaired):
        expected = "human" if index % 2 == 0 else "gpt"
        if message["from"] != expected:
            return None, True

    return repaired, changed
```

处理逻辑：

1. 把 `user/患者/病人` 统一成 `human`。
2. 把 `assistant/doctor/医生` 统一成 `gpt`。
3. 连续相同角色合并，避免出现两个 user 连续轮次。
4. 删除开头孤立 assistant。
5. 删除末尾没有回答的 user。
6. 最终验证 `human/gpt` 严格交替。

为什么末尾必须是 gpt：

> SFT 需要完整监督目标。如果最后只有用户问题，没有 assistant 答案，就没有可学习的 target。

## 2.7 SHA256 精确去重

真实代码：

```python
def compact_for_dedup(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def dedup_key_from_pair(question: str, answer: str) -> str:
    text = compact_for_dedup(question) + "\n" + compact_for_dedup(answer)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

这里不是只按问题去重，而是按“问题 + 答案”去重：

- 同一个问题不同答案可以保留。
- 问题和答案完全重复时只保留一条。
- 去掉空白和大小写差异，减少伪重复。

为什么使用哈希：

> 哈希 key 长度固定，放进 `set` 查询的平均复杂度是 O(1)，比保存和比较完整长文本更方便。

局限：

- 只能发现精确或规范化后的重复。
- “高血压怎么办”和“请问高血压该如何处理”仍可能是语义重复。
- 可以进一步使用 MinHash、SimHash 或 embedding 聚类做近重复去重。

## 2.8 清洗主流程怎么回答

30 秒回答：

> 清洗脚本流式遍历 JSONL，先解析字段并做 Unicode、HTML、控制字符和空白规范化，再根据问题和回答长度过滤异常样本，使用关键词和正则过滤广告联系方式。对于多轮数据，我会统一角色、合并连续同角色消息，并保证 human/gpt 交替。最后对规范化后的问题和答案计算 SHA256 做精确去重，同时用 Counter 统计每种过滤原因。50 万候选最终保留 381621 条。

源码定位题：

| 面试官问题 | 代码位置 |
| --- | --- |
| 文本在哪里规范化？ | `scripts/clean_corpus.py::normalize_text()` |
| 广告在哪里过滤？ | `contains_ad_or_contact()` |
| 角色在哪里修复？ | `repair_conversations()` |
| 去重 key 怎么生成？ | `dedup_key_from_pair()` |
| 主循环在哪里？ | `clean_corpus()` |

---

## 三、C-Eval 目标域向量筛选源码精读

## 3.1 为什么不是随机选 10 万

项目问题是：直接使用大规模通用医疗语料 SFT，不一定能提升目标评测，甚至可能因为分布不匹配导致 C-Eval 医学能力下降。

因此使用 C-Eval 的临床医学和基础医学题干描述目标域，再从清洗语料中筛选语义接近的样本。

注意数据泄漏边界：

> 主实验使用 `ceval_medical_question_only.jsonl`。C-Eval 题干只用于定义目标域分布，正确答案和标签不能作为 SFT target 混入训练集。否则评测结果会受到污染。

即使只使用题干，也需要在报告中明确这是 target-aware data selection，因为它可能让数据分布更贴近 benchmark。

## 3.2 BGE 编码实现

真实代码：

```python
class TransformersEmbedder:
    def __init__(self, model_name: str, device: str, max_length: int) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def encode(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts
            embeddings = normalize_rows(embeddings)
        return embeddings.cpu()
```

张量形状：

```text
input_ids:         [batch, sequence_length]
last_hidden_state: [batch, sequence_length, hidden_size]
attention_mask:    [batch, sequence_length, 1]
summed:            [batch, hidden_size]
embeddings:        [batch, hidden_size]
```

mean pooling：

```python
summed = (token_embeddings * attention_mask).sum(dim=1)
counts = attention_mask.sum(dim=1).clamp(min=1e-9)
embeddings = summed / counts
```

解释：

- padding token 的 mask 为 0，不进入求和。
- 有效 token hidden state 求和后除以有效 token 数。
- `clamp(min=1e-9)` 防止空文本导致除零。
- `torch.no_grad()` 因为这里只做推理，不保存反向图。
- `model.eval()` 关闭 dropout，保证结果稳定。

需要诚实说明：

> 当前实现使用通用 mean pooling。对于 BGE，生产级实现还应核对该模型官方推荐的 pooling、query instruction 和归一化方式。当前流程是可复现的领域相似筛选实现，但不是对所有 embedding 模型都最优的统一方案。

## 3.3 为什么归一化后点积等于余弦相似度

真实代码：

```python
def normalize_rows(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)
```

余弦相似度：

```text
cos(x, y) = (x · y) / (||x||₂ ||y||₂)
```

归一化后：

```text
x' = x / ||x||₂
y' = y / ||y||₂
x' · y' = cos(x, y)
```

所以可以直接矩阵乘法：

```python
score_matrix = candidate_embeddings @ target_embeddings.T
best_scores, best_indices = score_matrix.max(dim=1)
```

假设：

```text
candidate_embeddings: [64, 384]
target_embeddings:    [426, 384]
target_embeddings.T:  [384, 426]
score_matrix:         [64, 426]
```

每一行表示一条候选语料与 426 个医学目标的相似度。`max(dim=1)` 取最相似目标。

为什么取最大值而不是平均值：

> 一条样本通常只属于某个局部医学主题。如果与某一道临床或基础医学目标高度相关，就有保留价值。对全部目标求平均会稀释这种局部相关性。它的缺点是容易被单个偶然高分影响，可以改进为 Top-N 均值、按科目加权或加入质量分。

## 3.4 候选文本如何构造

真实代码：

```python
def build_candidate_text(record: dict[str, Any], template: str) -> str:
    return template.format(
        instruction=str(record.get("instruction", "")).strip(),
        input=str(record.get("input", "")).strip(),
        output=str(record.get("output", "")).strip(),
    ).strip()
```

配置模板：

```text
指令：{instruction}
问题：{input}
回答：{output}
```

为什么把回答也编码：

> 目标是筛选高质量 SFT 对，不只是相似问题。把回答放入 embedding 可以利用答案中的医学知识点。但它也可能偏向措辞与目标题干相似的答案。更严谨的对照实验应分别比较 question-only 与 question-answer candidate embedding。

## 3.5 最小堆维护 Top 100000

真实代码：

```python
def update_heap(
    heap: list[tuple[float, int, dict[str, Any]]],
    top_k: int,
    record: dict[str, Any],
    score: float,
    sequence_id: int,
) -> None:
    item = (score, sequence_id, record)
    if len(heap) < top_k:
        heapq.heappush(heap, item)
        return
    if score > heap[0][0]:
        heapq.heapreplace(heap, item)
```

为什么是最小堆：

- 堆顶 `heap[0]` 是当前 Top-K 中最低分。
- 新样本不超过堆顶就丢弃。
- 新样本更高时替换堆顶。
- 每次更新复杂度 O(log K)。
- 不需要保存 381621 条完整结果后再全部排序。

`sequence_id` 的作用：

> Python tuple 比较会继续比较后续元素。两个分数相同时，如果直接放字典，字典不可排序。加入递增整数作为第二关键字，可以稳定打破平局。

整体复杂度：

```text
编码：主要由 embedding 模型前向决定
相似度：O(N × M × D)
Top-K：O(N log K)
堆内存：O(K)
```

其中：

- N = 381621 条候选
- M = 426 条目标
- D = embedding 维度
- K = 100000

## 3.6 相似筛选标准回答

2 分钟回答：

> 我先读取 C-Eval 临床医学和基础医学的不带答案目标集，共 426 条，使用 BAAI/bge-small-zh-v1.5 编码并做 L2 归一化。候选 SFT 样本按 instruction、input、output 拼成文本，分 batch 编码。归一化后，候选矩阵乘以目标矩阵转置就得到余弦相似度矩阵，每条候选取最大相似度以及对应的目标题目。为了不用保存并全排序 38 万条结果，我使用容量 10 万的最小堆，堆顶始终是当前入选样本最低分。最终入选 10 万条，最低分约 0.6192。这里 C-Eval 只用于目标域分布参考，主实验没有把答案作为训练标签。

高频追问：这能叫“高质量”吗？

> 更准确地说，它首先保证目标域相关性，不等同于完整的数据质量。质量还来自前置清洗。后续可以增加事实正确性模型、重复度、多样性和人工抽检，把相关性分与质量分联合排序。

---

## 四、SFT 数据进入 Qwen3 的完整过程

## 4.1 SFT 在学什么

给定 prompt token `x` 和标准答案 token `y`，自回归模型学习：

```text
P(y | x) = ∏ P(y_t | x, y_<t)
```

teacher forcing 表示训练第 `t` 个答案 token 时，模型看到的是标准答案前缀，而不是自己刚才生成的 token。

交叉熵可以写成：

```text
L = - Σ log P(y_t | x, y_<t)
```

本项目不是让模型复现用户问题，而是学习 assistant 回答，所以用户 prompt 对应位置的 label 必须 mask。

## 4.2 ShareGPT 数据如何进入 Qwen3 template

训练数据示例：

```json
{
  "conversations": [
    {"from": "human", "value": "糖尿病患者如何控制血糖？"},
    {"from": "gpt", "value": "需要结合饮食、运动、监测和规范用药……"}
  ]
}
```

MedicalGPT 的 `get_dialog()` 会把角色转换为标准消息：

```python
if role in ["human", "user", "observation"]:
    messages.append({"role": "user", "content": value})
elif role in ["gpt", "assistant", "function_call"]:
    messages.append({"role": "assistant", "content": value})
```

当指定 `--template_name qwen3` 时：

```python
if prompt_template:
    yield prompt_template.get_dialog(
        history_messages,
        system_prompt=system_prompt
    )
```

否则使用 tokenizer 自带模板：

```python
cur_text = tokenizer.apply_chat_template(
    accumulated,
    tokenize=False,
    add_generation_prompt=True,
)
```

这里的 `template_name` 不是选择模型权重。模型权重由：

```text
--model_name_or_path Qwen/Qwen3-4B-Instruct-2507
```

决定；`--template_name qwen3` 只决定对话文本如何包装成 Qwen3 能识别的 prompt。

## 4.3 最重要的 labels mask

真实训练代码：

```python
for i in range(len(dialog) // 2):
    source_ids = tokenizer.encode(
        text=dialog[2 * i],
        add_special_tokens=(i == 0),
    )
    target_ids = tokenizer.encode(
        text=dialog[2 * i + 1],
        add_special_tokens=False,
    )

    input_ids += source_ids + target_ids + [tokenizer.eos_token_id]
    if script_args.train_on_inputs:
        labels += source_ids + target_ids + [tokenizer.eos_token_id]
    else:
        labels += (
            [IGNORE_INDEX] * len(source_ids)
            + target_ids
            + [tokenizer.eos_token_id]
        )
```

假设 token 是：

```text
source_ids = [10, 11, 12]
target_ids = [20, 21]
eos = 2
```

那么：

```text
input_ids = [10, 11, 12, 20, 21, 2]
labels    = [-100, -100, -100, 20, 21, 2]
```

`CrossEntropyLoss` 默认忽略 label 为 `-100` 的位置，因此：

- 用户 prompt 仍是模型的上下文。
- prompt token 不贡献监督 loss。
- assistant answer 和 EOS 参与 loss。

面试标准回答：

> 我使用 causal LM 的 teacher forcing 做 SFT。输入中同时包含用户和 assistant token，但默认 `train_on_inputs=False`，用户部分 labels 设置成 `IGNORE_INDEX=-100`，只有 assistant 回答和 EOS 参与交叉熵。这能避免模型把训练目标浪费在复述用户输入上。

追问：为什么 attention mask 不能代替 label mask？

> attention mask 控制 token 是否能被注意力看到，padding 通常为 0。用户 prompt 必须被 assistant 看到，所以它的 attention mask 应为 1；只是不能参与 loss，因此要使用 label mask。

## 4.4 截断策略

真实代码按当前 source 和 target 比例分配最大长度：

```python
total_len = len(source_ids) + len(target_ids)
max_source_len = int(max_length * (len(source_ids) / total_len))
max_target_len = int(max_length * (len(target_ids) / total_len))

if len(source_ids) > max_source_len:
    source_ids = source_ids[:max_source_len]
if len(target_ids) > max_target_len - 1:
    target_ids = target_ids[:max_target_len - 1]
```

优点：

- 避免 source 或 target 单方面占满上下文。
- 为 EOS 留一个 token。

局限：

- 对医学长回答，按比例截断可能丢失结尾的风险提示。
- 更好的方案可以设置最小 answer token 配额、按轮次截断、保留回答结尾，或使用更长上下文。

## 4.5 Trainer 做了什么

项目使用 `SavePeftModelTrainer`，其基础仍是 Transformers `Trainer`。训练器负责：

1. DataLoader 取 batch。
2. data collator 对 `input_ids/attention_mask/labels` padding。
3. 模型 forward 返回 loss。
4. loss 除以梯度累积步数。
5. backward。
6. 达到 accumulation steps 后 optimizer step。
7. scheduler step。
8. 清零梯度。
9. 日志、验证和 checkpoint 保存。

你不需要从零背 Trainer 源码，但必须知道这条调用链。

---

## 五、LoRA / QLoRA 原理与实操参数

## 5.1 LoRA 数学原理

对原始线性层：

```text
h = Wx
```

LoRA 不直接更新 `W`，而是增加低秩分支：

```text
h = Wx + sBAx
s = alpha / r
```

假设 `W ∈ R^(d_out × d_in)`：

```text
A ∈ R^(r × d_in)
B ∈ R^(d_out × r)
```

当 `r` 远小于 `d_in` 和 `d_out` 时，训练参数从：

```text
d_out × d_in
```

降为：

```text
r × d_in + d_out × r
```

## 5.2 LoRA 如何初始化

常见实现是：

- A 使用随机初始化。
- B 初始化为 0。

因此训练刚开始时：

```text
BA = 0
```

模型初始行为与 base model 基本一致，之后逐渐学习增量。具体初始化细节由 PEFT 版本和配置实现，项目没有手写 A/B 初始化。

面试不能说：

> “A、B 都是我在训练脚本中手动初始化的。”

正确说法：

> LoRA adapter 的底层注入和初始化由 PEFT 完成，我配置 rank、alpha、dropout 和 target modules，并理解常见的 A 随机、B 零初始化设计。

## 5.3 QLoRA 的真实量化配置

项目代码：

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
```

含义：

- `load_in_4bit=True`：base model 权重以 4bit 加载。
- `nf4`：NormalFloat4，针对近似正态分布权重设计的 4bit 数据类型。
- `double_quant=True`：对量化尺度再次量化，进一步节省显存。
- `compute_dtype=bfloat16`：权重存储是 4bit，但矩阵计算使用 bf16。

重要区别：

```text
4bit 是 base 权重的存储表示。
LoRA 可训练参数通常仍使用较高精度。
QLoRA 不是把所有训练计算都变成 4bit。
```

## 5.4 真实 LoRA 配置

```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules,
    inference_mode=False,
    r=script_args.lora_rank,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    modules_to_save=modules_to_save,
)
model = get_peft_model(model, peft_config)
```

脚本默认参数：

```text
r = 8
lora_alpha = 32
lora_dropout = 0.05
target_modules = all
```

当 target modules 为 `all` 时，真实查找函数：

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
            if "lm_head" in name:
                continue
            if "output_layer" in name:
                continue
            names = name.split(".")
            lora_module_names.add(
                names[0] if len(names) == 1 else names[-1]
            )
    return sorted(lora_module_names)
```

为什么量化后查找 `Linear4bit`：

> QLoRA 加载后，原始 `torch.nn.Linear` 会被 bitsandbytes 的 `Linear4bit` 替换。如果仍然只查 `torch.nn.Linear`，会漏掉需要注入 LoRA 的层。

为什么排除 `lm_head`：

> 输出词表层参数量大且直接控制 token logits，通常不作为默认 LoRA target。是否训练它取决于任务和 tokenizer 是否变化。

## 5.5 LoRA 超参数怎么解释

### rank `r`

- 越大，增量矩阵表达能力越强。
- 可训练参数、显存和过拟合风险增加。
- 效果差不一定直接增加 rank，应先检查数据和学习率。

### `lora_alpha`

缩放通常与 `alpha/r` 有关。它控制 LoRA 分支影响强度，不能脱离 rank 单独比较。

### `lora_dropout`

- 训练时随机丢弃 LoRA 分支输入。
- 小数据时可缓解过拟合。
- 过大可能导致欠拟合。

### target modules

Qwen 类模型常见：

```text
注意力：q_proj, k_proj, v_proj, o_proj
FFN：gate_proj, up_proj, down_proj
```

只训练 q/v 更省参数；覆盖所有线性层通常表达能力更强，但成本也更高。

### learning rate

LoRA 学习率通常可以高于全参微调，但需要结合 batch、数据量和 rank。出现 loss 震荡、遗忘或输出崩坏时应降低。

## 5.6 有效 batch size

```text
effective batch size
= per_device_train_batch_size
× gradient_accumulation_steps
× data_parallel_world_size
```

例如单卡：

```text
1 × 8 × 1 = 8
```

梯度累积不降低单次 forward 的激活显存，但可以在小 micro-batch 下模拟更大的更新 batch。

## 5.7 LoRA 效果不好怎么排查

推荐顺序：

1. 检查数据格式和 labels mask，不要先盲目加 rank。
2. 查看训练 loss、验证 loss 和生成样例。
3. 确认 target modules 真正被注入。
4. 检查 trainable parameter 数量。
5. 调整学习率、warmup、epoch 和 max length。
6. 增加 rank 或覆盖更多线性层。
7. 检查 4bit 量化是否造成明显损失，做 LoRA 与 QLoRA 对照。
8. 检查训练数据与评测域是否匹配。

## 5.8 LoRA 过拟合怎么解决

- 减少 epoch 或提前停止。
- 降低 rank、alpha 或学习率。
- 增加 dropout。
- 扩充和去重数据。
- 单独保留验证集，不要训练和验证使用完全相同文件。
- 降低重复样本和模板化回答比例。
- 监控通用能力，防止领域灾难性遗忘。

## 5.9 LoRA 的缺点

- 低秩假设不一定覆盖所有任务所需更新。
- adapter 与 base model 版本必须匹配。
- 多 adapter 管理和部署更复杂。
- merge 后失去灵活切换 adapter 的能力。
- rank 太低可能欠拟合，太高会增加成本。
- QLoRA 的量化误差可能限制上限。

---

## 六、PPO 多维奖励与四组件实现

## 6.1 PPO 数据为什么不能直接等同于 SFT 数据

SFT 数据：

```json
{
  "instruction": "请回答以下问题",
  "input": "病例问题",
  "output": "标准答案"
}
```

PPO 数据：

```json
{
  "prompt": "病例问题",
  "reference_answer": "参考答案",
  "answer_keywords": ["关键词1", "关键词2"],
  "risk_level": "high",
  "required_sections": [
    "病情分析",
    "处理建议",
    "风险提示",
    "就医建议"
  ]
}
```

SFT 用 `output` 直接算 token 级交叉熵；PPO 让 policy 在线生成回答，再根据生成结果计算 sequence reward。

## 6.2 5K PPO 数据如何构造

真实转换函数：

```python
def convert_row(row: dict[str, Any], row_id: int) -> dict[str, Any] | None:
    prompt = build_prompt(row)
    reference_answer = clean_text(row.get("output"))
    if not prompt or not reference_answer:
        return None

    keywords = candidate_terms(reference_answer)
    if not keywords:
        keywords = candidate_terms(prompt)
    if not keywords:
        return None

    risk_level = (
        "high"
        if is_high_risk(prompt, reference_answer)
        else "normal"
    )
    return {
        "id": f"medical-ppo-{row_id}",
        "prompt": prompt,
        "reference_answer": reference_answer,
        "answer_keywords": keywords,
        "risk_level": risk_level,
        "required_sections": DEFAULT_REQUIRED_SECTIONS,
        "source": row.get("source", "shibing624/medical"),
        "source_row_id": row.get("row_id", row_id),
        "similarity_score": row.get("similarity_score"),
        "best_target_id": row.get("best_target_id"),
    }
```

数据关联：

```text
SFT Top100k 的 instruction + input -> PPO prompt
SFT Top100k 的 output -> reference_answer
reference_answer -> answer_keywords
prompt + answer -> risk_level
```

因此 SFT 与 RL 数据同源，但用途不同：

- SFT 模仿参考回答。
- PPO 根据 policy 当前输出与规则奖励做在线优化。

## 6.3 为什么保证高风险样本比例

真实抽样代码：

```python
high_risk = [r for r in records if r["risk_level"] == "high"]
normal = [r for r in records if r["risk_level"] != "high"]
rng = random.Random(seed)
rng.shuffle(high_risk)
rng.shuffle(normal)

min_high = min(len(high_risk), int(sample_size * 0.30))
selected = high_risk[:min_high]
selected.extend(normal[: sample_size - len(selected)])
```

如果高风险病例太少：

- 大部分样本的安全分恒为高分。
- policy 很难学到高风险场景下的就医提示。
- safety reward 对梯度贡献不足。

当前规则目标是至少约 30% 高风险样本，但实际能否达到还取决于候选池数量。

## 6.4 三维奖励真实代码

### 格式分

```python
def compute_format_score(
    response: str,
    required_sections: list[str] | None = None,
) -> float:
    response = normalize_text(response)
    sections = required_sections or DEFAULT_REQUIRED_SECTIONS
    if not response:
        return 0.0
    hits = 0
    for section in sections:
        section = normalize_text(section)
        if section and section in response:
            hits += 1
    return hits / max(len(sections), 1)
```

四个标题每命中一个得 0.25。它衡量结构合规，不衡量医学事实正确性。

### 关键词分

```python
def compute_keyword_score(
    response: str,
    answer_keywords: list[str] | None = None,
) -> float:
    response = normalize_text(response).lower()
    keywords = [
        normalize_text(k).lower()
        for k in (answer_keywords or [])
        if normalize_text(k)
    ]
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in response)
    return hits / len(keywords)
```

### 字符 F1

```python
def char_f1(prediction: str, reference: str) -> float:
    pred_chars = [
        c for c in normalize_text(prediction)
        if not c.isspace()
    ]
    ref_chars = [
        c for c in normalize_text(reference)
        if not c.isspace()
    ]
    if not pred_chars or not ref_chars:
        return 0.0

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for char in pred_chars:
        pred_counts[char] = pred_counts.get(char, 0) + 1
    for char in ref_chars:
        ref_counts[char] = ref_counts.get(char, 0) + 1

    overlap = sum(
        min(pred_counts.get(char, 0), count)
        for char, count in ref_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_chars)
    recall = overlap / len(ref_chars)
    return 2 * precision * recall / (precision + recall)
```

准确率代理：

```python
def compute_accuracy_score(
    response: str,
    reference_answer: str | None = None,
    answer_keywords: list[str] | None = None,
) -> float:
    keyword_score = compute_keyword_score(response, answer_keywords)
    semantic_proxy = char_f1(response, reference_answer or "")
    if answer_keywords:
        return 0.7 * keyword_score + 0.3 * semantic_proxy
    return semantic_proxy
```

这里必须准确表述：

> 代码里的 `semantic_proxy` 实际是字符级 F1，不是真正的 embedding 语义相似度。它可以衡量字面重合，但对同义改写不敏感。如果简历写“语义相似度”，需要补充 BGE 或 cross-encoder 实现，否则面试时应该称为“关键词覆盖率与字符 F1 代理分”。

### 安全分

```python
def compute_safety_score(
    prompt: str,
    response: str,
    risk_level: str = "normal",
) -> float:
    prompt = normalize_text(prompt)
    response = normalize_text(response)
    text = f"{prompt}\n{response}"
    score = 1.0

    for term in SAFETY_NEGATIVE_TERMS:
        if term in response:
            score -= 0.25

    is_high = (
        risk_level == "high"
        or any(term in prompt for term in HIGH_RISK_PROMPT_TERMS)
    )
    if is_high and not any(
        term in response for term in SAFETY_POSITIVE_TERMS
    ):
        score -= 0.35
    if any(term in response for term in SAFETY_POSITIVE_TERMS):
        score += 0.10
    if "诊断" in text and "医生" not in response and "检查" not in response:
        score -= 0.10
    return max(0.0, min(1.0, score))
```

设计含义：

- 出现“自行停药”“保证治愈”等危险表达扣分。
- 高风险病例缺少就医或医生提示扣分。
- 出现合理就医提示加分。
- 最终裁剪到 `[0, 1]`。

局限：

- 关键词规则可能误判否定句。
- 模型可能堆砌“及时就医”骗取安全分。
- 安全不只是是否出现固定词。
- 后续可以加入医疗安全分类器、LLM judge 和人工复核。

### 总奖励

```python
def compute_total_reward(
    record: dict[str, Any],
    response: str,
) -> dict[str, float]:
    format_score = compute_format_score(
        response,
        record.get("required_sections"),
    )
    accuracy_score = compute_accuracy_score(
        response,
        reference_answer=record.get("reference_answer"),
        answer_keywords=record.get("answer_keywords"),
    )
    safety_score = compute_safety_score(
        record.get("prompt", ""),
        response,
        record.get("risk_level", "normal"),
    )
    total = (
        FORMAT_WEIGHT * format_score
        + ACCURACY_WEIGHT * accuracy_score
        + SAFETY_WEIGHT * safety_score
    )
    return {
        "format": float(format_score),
        "accuracy": float(accuracy_score),
        "safety": float(safety_score),
        "total": float(total),
    }
```

公式：

```text
R = 0.30 R_format + 0.50 R_accuracy + 0.20 R_safety
```

为什么准确率权重最高：

> 医疗回答首先要尽量覆盖正确医学要点。格式只是可读性约束，安全是底线约束。第一版采用 0.5/0.3/0.2 是工程先验，严格来说需要通过消融实验比较不同权重，而不能声称它是理论最优值。

## 6.5 PPO 四类组件

| 组件 | 本项目实现 |
| --- | --- |
| Policy | SFT 后 Qwen3，参与更新并生成回答 |
| Reference | 冻结的 SFT 后 Qwen3，用于 KL 约束 |
| Value | Qwen3 backbone + `ValueScoreHead` |
| Reward | `RuleBasedRewardModel`，内部调用手写奖励函数 |

PPO 不是必须有四个不同架构、四份完整模型文件。“四个模型”更准确地说是四类功能组件。

## 6.6 为什么规则奖励仍然可以做 PPO

PPO 需要的是每个 rollout 的标量 reward：

```text
response -> reward scalar
```

这个 reward 可以来自：

- 人工训练的 reward model。
- 可验证的正确答案。
- 编译器或单元测试。
- 数学 verifier。
- 手写规则函数。

因此没有神经网络 reward model 不代表不是 PPO。是否属于 PPO，取决于 policy 是否根据 rollout、advantage、value、KL 和 clipped objective 做更新。

更准确的项目表述：

> 我没有额外训练神经网络 reward model，而是把医疗格式、答案覆盖和安全约束写成规则奖励，通过 TRL-compatible wrapper 接入 PPOTrainer。

## 6.7 RuleBasedRewardModel 为什么要包装

TRL 的 `get_reward()` 期望 reward model 具有：

- `base_model_prefix`
- 对应的 backbone
- `.score(hidden_states)` 头

真实 wrapper：

```python
class RuleBasedRewardModel(torch.nn.Module):
    base_model_prefix = "backbone"

    def __init__(
        self,
        tokenizer: Any,
        prompt_index: list[tuple[str, str, dict[str, Any]]],
        pad_token_id: int,
    ):
        super().__init__()
        self.backbone = TokenPassthroughBackbone()
        self.score = RuleBasedScoreHead(
            tokenizer,
            prompt_index,
            pad_token_id,
        )
        self.config = SimpleNamespace(
            model_type="rule_based_medical_reward"
        )
```

`RuleBasedScoreHead` 会：

1. 从 token id 解码完整文本。
2. 匹配原始 prompt 对应的数据记录。
3. 截出 response。
4. 调用 `compute_total_reward()`。
5. 把标量 reward 扩展成 TRL 预期的形状。

这层不是一个学出来的 reward network，而是接口适配器。

## 6.8 ValueModelWrapper 为什么存在

普通 `Qwen3ForCausalLM` 没有 value head，也没有 `.score`：

```python
class ValueScoreHead(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self.proj.weight.device != hidden_states.device
            or self.proj.weight.dtype != hidden_states.dtype
        ):
            self.proj.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        return self.proj(hidden_states)
```

```python
class ValueModelWrapper(torch.nn.Module):
    base_model_prefix = "pretrained_model"

    def __init__(self, pretrained_model: torch.nn.Module):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.config, "text_config"):
            hidden_size = getattr(
                self.config.text_config,
                "hidden_size",
                None,
            )
        if hidden_size is None:
            raise ValueError(
                "Cannot infer hidden_size for PPO value head."
            )
        self.score = ValueScoreHead(hidden_size)

    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        return self.pretrained_model(*args, **kwargs)
```

value head 为每个 token 的 hidden state 预测标量价值，用于估计 advantage。

`dtype/device` 动态同步解决了实际遇到的：

```text
BFloat16 and Float dtype mismatch
```

## 6.9 PPOTrainer 接口

```python
trainer = PPOTrainer(
    args=ppo_args,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_model,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model(ppo_args.output_dir)
```

TRL 内部大致流程：

```text
从 prompt dataset 取 batch
-> policy 生成 response
-> policy/ref 计算 token log probabilities
-> reward function 产生 sequence reward
-> 加入 reference KL penalty
-> value model 估计 value
-> 计算 return 与 advantage
-> 对同一 rollout 做若干 PPO epoch
-> clipped policy loss + value loss + 其他正则
-> 更新 policy 和 value
```

PPO clipped objective 的直观形式：

```text
r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)

L_clip = E[min(
    r_t A_t,
    clip(r_t, 1-ε, 1+ε) A_t
)]
```

裁剪的目的：

> 防止一次更新把新 policy 推得离采样时的 old policy 太远，提高训练稳定性。reference KL 则约束模型不要偏离 SFT 模型过多，两者不是同一个约束。

## 6.10 Reward hacking 风险

当前奖励可能被以下方式利用：

- 重复输出四个标题刷格式分。
- 堆砌关键词但医学逻辑错误。
- 每个回答都加“及时就医”刷安全分。
- 复制参考答案获得高字符 F1。
- 生成很长回答提高关键词命中概率。

改进方案：

- 设置标题只计一次并检查段落是否非空。
- 加长度惩罚和重复惩罚。
- 高风险与普通风险采用不同安全规则。
- 用 embedding/cross-encoder 或医学 LLM judge 评估事实。
- 加入 hard negative 和对抗样本。
- 做各奖励项的消融实验。
- 监控 reward、KL、response length、entropy 和各子奖励分布。

---

## 七、PPL、C-Eval 与格式准确率评测

## 7.1 C-Eval

C-Eval 医学子任务：

```text
basic_medicine
clinical_medicine
```

项目通过 lm-evaluation-harness 统一评测 base、SFT 和 PPO 模型。

最终总结采用的结果：

```text
Base: 0.6902
SFT:  0.7620
PPO:  0.7711
```

正确解读：

- SFT 是主要提升来源。
- PPO 相对 SFT 只有小幅趋势提升。
- 医学子任务样本量有限，不能把小幅差异夸大成统计显著。

## 7.2 answer-only PPL 为什么合理

评测目标是：给定医疗问题后，模型对参考答案的建模能力。

因此 prompt 作为上下文，但不计入 loss。

真实代码：

```python
def build_answer_only_features(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompt_text = render_prompt(
        tokenizer,
        build_user_prompt(row),
    )
    answer_text = clean_text(row.get("output"))
    if tokenizer.eos_token:
        answer_text += tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    ).input_ids
    answer_ids = tokenizer(
        answer_text,
        add_special_tokens=False,
    ).input_ids

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
        len(answer_ids),
    )
```

PPL 定义：

```text
NLL = 所有答案 token 的负对数似然均值
PPL = exp(NLL)
```

真实聚合：

```python
total_nll += float(loss.detach().cpu()) * answer_tokens
total_answer_tokens += answer_tokens

eval_loss = total_nll / total_answer_tokens
perplexity = math.exp(eval_loss)
```

为什么不能直接平均每条样本 loss：

> 每条答案 token 数不同。直接对样本 loss 等权平均会让短答案和长答案权重相同。先乘 token 数累加，再除总 token 数，得到真正的 token-level NLL。

真实结果：

```text
Base:
eval_loss = 2.5122
PPL = 12.3325

SFT:
eval_loss = 2.2856
PPL = 9.8318
```

同一评测集包含 1000 条、共 1924264 个答案 token，因此结果可直接对比。

PPL 局限：

- 低 PPL 不保证事实正确。
- 如果评测集来自训练数据分布，可能高估领域适配。
- 量化方式、chat template 和截断必须保持一致。
- 不同 tokenizer 的 PPL 不能直接横向比较。

## 7.3 复杂病例格式准确率

真实代码：

```python
def compute_format_score(
    response: str,
    required_sections: list[str] | None = None,
) -> tuple[float, list[str]]:
    response = normalize_text(response)
    sections = required_sections or DEFAULT_REQUIRED_SECTIONS
    missing = [
        section
        for section in sections
        if normalize_text(section)
        and normalize_text(section) not in response
    ]
    hits = len(sections) - len(missing)
    return hits / max(len(sections), 1), missing
```

通过条件：

```python
cur_format_pass = cur_format_score == 1.0
```

主指标：

```text
format_accuracy = 四个标题全部命中的样本数 / 总样本数
```

辅助指标：

- `avg_format_score`：平均命中标题比例。
- `safety_coverage`：安全规则通过比例。
- `keyword_coverage`：参考关键词覆盖比例。

## 7.4 prompted 与 unprompted 必须区分

如果 prompt 明确写：

```text
请严格按照以下四个小标题回答：
1. 病情分析
2. 处理建议
3. 风险提示
4. 就医建议
```

测到的是：

```text
带格式提示下的格式指令遵循准确率
prompted format accuracy
```

如果 prompt 没有格式要求，测到的是：

```text
无格式提示下的自发结构化输出率
unprompted format accuracy
```

两者都合理，但不能混称。最严谨的是同时报告两列。

简历中的 `72% -> 94%` 应准备以下证据：

- 固定评测样本。
- 相同生成参数。
- SFT/PPO 相同 prompt。
- 明确是否带格式提示。
- 自动脚本或人工抽检规则。
- 对应 report 或原始 responses。

---

## 八、训练框架调用关系与必须掌握的 API

## 8.1 Transformers

必须知道：

```python
AutoTokenizer.from_pretrained(...)
AutoModelForCausalLM.from_pretrained(...)
BitsAndBytesConfig(...)
TrainingArguments / Seq2SeqTrainingArguments
Trainer
DataCollatorForSeq2Seq
model.generate(...)
model.save_pretrained(...)
```

职责：

- 加载 tokenizer 和 Qwen3 causal LM。
- chat template 与 tokenization。
- 标准 SFT 训练循环。
- 生成、保存、checkpoint 和日志。

面试回答：

> Transformers 是基础模型训练和推理层。我的 SFT 使用它加载 Qwen3、构建 tokenizer、Trainer 和 data collator；PPL 和格式评测也使用它加载模型和执行 forward/generate。

## 8.2 PEFT

必须知道：

```python
LoraConfig(...)
get_peft_model(...)
prepare_model_for_kbit_training(...)
PeftModel.from_pretrained(...)
merge_and_unload()
```

职责：

- 注入 LoRA adapter。
- 冻结 base 参数。
- 让量化模型适合 k-bit 训练。
- 加载、保存和合并 adapter。

## 8.3 bitsandbytes

必须知道：

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

职责：

- 用 4bit 加载 base model。
- 提供 `Linear4bit`。
- 降低 QLoRA 显存占用。

## 8.4 Datasets

必须知道：

```python
load_dataset(...)
Dataset.from_list(...)
dataset.map(...)
dataset.filter(...)
dataset.shuffle(...)
dataset.select(...)
train_test_split(...)
```

职责：

- 读取本地或 Hub 数据。
- 批量预处理和过滤。
- 构造训练/验证切分。

## 8.5 TRL

项目实操重点：

```python
PPOConfig
PPOTrainer
```

概念了解但不声称项目主线实操：

```text
DPOTrainer
GRPOTrainer
RLOOTrainer
RewardTrainer
```

必须知道 PPOTrainer 需要：

- policy
- reference
- value
- reward
- tokenizer/processing class
- prompt dataset
- PPO config

## 8.6 SwanLab

训练参数：

```text
--report_to swanlab
--run_name ...
```

Transformers/Trainer 可自动记录：

- training loss
- eval loss
- learning rate
- epoch/step
- gradient norm
- runtime 和 samples per second
- 训练参数配置

自定义 reward 子指标如果没有显式 `swanlab.log()` 或进入 Trainer metrics，不一定会自动展示。不能只加 `report_to` 就声称所有自定义分数都记录了。

## 8.7 lm-evaluation-harness

必须知道命令组成：

```text
--model hf
--model_args pretrained=...,peft=...,dtype=...
--tasks ...
--device cuda:0
--batch_size ...
--output_path ...
```

它负责：

- 加载模型。
- 构造 benchmark prompt。
- 计算选项 log-likelihood 或任务指标。
- 汇总准确率和标准误。

## 8.8 adapter、checkpoint 和 merged model

### LoRA adapter

通常包含：

```text
adapter_config.json
adapter_model.safetensors
```

加载方式：

```python
base = AutoModelForCausalLM.from_pretrained(base_path)
model = PeftModel.from_pretrained(base, adapter_path)
```

### 完整 checkpoint

包含完整模型权重和配置，可以：

```python
AutoModelForCausalLM.from_pretrained(checkpoint_path)
```

不能因为目录叫 `checkpoint-300` 就默认它一定是 adapter，必须看文件内容。

### merged model

通过：

```python
merged = peft_model.merge_and_unload()
merged.save_pretrained(output_path)
```

优点：

- 推理部署简单。
- 不需要额外加载 adapter。

缺点：

- 文件大。
- 不能轻易切换 adapter。
- 后续继续 LoRA 训练时要明确从 merged model 还是 base + adapter 开始。

---

## 九、训练故障与排查经验

## 9.1 参数解析失败

错误：

```text
Some specified arguments are not used by HfArgumentParser:
['--overwrite_output_dir']
```

原因：

- 当前脚本使用的 dataclass 参数中没有暴露该字段，或 Transformers 版本与脚本接口不一致。

处理：

- 先查看 `--help` 和 dataclass。
- 删除未被当前脚本接受的参数。
- 不要盲目照抄其他版本命令。

面试价值：

> 我学到训练脚本的命令行参数由 `HfArgumentParser` 对应的 dataclass 决定，框架版本升级后参数接口可能变化，因此先核对本地源码，而不是只依赖 README。

## 9.2 PPO value model 缺少接口

错误：

```text
Qwen3ForCausalLM object has no attribute 'score'
```

原因：

- causal LM 输出 logits，不是 value/reward sequence classifier。
- TRL experimental PPO 期望 value model 暴露 `.score`。

解决：

- 增加 `ValueModelWrapper`。
- 强制输出 hidden states。
- 增加 `Linear(hidden_size, 1)` value head。

## 9.3 dtype mismatch

错误：

```text
BFloat16 and Float
```

原因：

- Qwen hidden states 是 bf16。
- 新建 value head 默认是 fp32。

解决代码：

```python
self.proj.to(
    device=hidden_states.device,
    dtype=hidden_states.dtype,
)
```

## 9.4 left padding

PPO 批量生成 prompt 时设置：

```python
tokenizer.padding_side = "left"
```

原因：

> 对 decoder-only 模型做批量生成时，左侧 padding 可以让不同长度 prompt 的最后一个有效 token 对齐，避免右侧 padding 后从 pad 位置继续生成。

## 9.5 OOM

优先排查：

- `max_prompt_length`
- `max_new_tokens`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- 是否 4bit
- 是否 gradient checkpointing
- PPO 是否同时加载 policy/ref/value 多份 4B 模型
- checkpoint 保存是否额外复制权重

不能把 gradient accumulation 当作降低单次激活显存的万能方案。真正降低单步峰值通常需要减小 micro-batch、序列长度或模型副本。

## 9.6 磁盘 100% 与 checkpoint 保存失败

处理顺序：

1. `df -h` 查看磁盘。
2. 检查 Hugging Face cache、训练 checkpoint、SwanLab 日志。
3. 删除残缺 checkpoint，而不是当作可恢复模型。
4. 设置 `save_total_limit`。
5. 必要时只保存模型，不保存 optimizer state。
6. 把 HF cache 指向大容量挂载盘。

## 9.7 复杂病例格式准确率为 0

根因：

- 模型回答了医学内容。
- prompt 没要求四段标题。
- evaluator 却严格检查四个固定字符串。

修复：

- 明确区分 prompted/unprompted 指标。
- 在数据 prompt 中加入格式要求。
- `max_new_tokens` 从 128 提升到 256 或 512，减少截断。
- 保存无提示数据作为对照。

这个问题说明：

> 评测协议必须和任务定义一致。不能要求模型完成 prompt 从未提出的格式，再把失败全部归因于模型能力。

---

## 十、高频面试问题标准答案

## 10.1 你在项目中亲手实现了什么？

30 秒：

> 我亲手实现的是数据和任务定制层，包括 50 万医疗语料准备、清洗和格式统一，C-Eval 医学目标集构建，BGE embedding 相似筛选与最小堆 Top-K，5K PPO 数据构造，格式、准确率、安全多维奖励，以及 PPL 和复杂病例格式评测。SFT、LoRA 注入和 PPO 优化器底层基于 MedicalGPT、Transformers、PEFT 和 TRL，我完成了 Qwen3 数据适配、参数配置、wrapper 扩展、训练调试和评测闭环。

## 10.2 为什么清洗后只剩 381621 条？

> 原始 50 万中，主要过滤来源是过短样本 115194 条，其次是精确重复 2783 条、广告联系方式 361 条和过长样本 41 条。最终保留 381621 条。每一种过滤原因都写入 cleaning report，方便复查规则是否过严。

## 10.3 为什么用 C-Eval 做数据筛选？

> 因为训练目标是提升临床医学和基础医学能力，原始医疗数据分布很宽。C-Eval 题干可以作为目标域 query，帮助筛出更接近目标医学知识分布的样本。但我没有把答案作为训练标签，主实验使用不带答案目标集，并把这种方法定位为 target-aware selection，而不是完全 benchmark-independent 的数据选择。

## 10.4 embedding 相似度代码怎么实现？

> 候选和目标用 BGE 编码并 L2 归一化，然后通过 `candidate_embeddings @ target_embeddings.T` 得到余弦相似矩阵。每条候选取最大相似度和目标 ID，再用容量 10 万的最小堆维护 Top-K，复杂度是相似计算 O(NMD)，堆更新 O(N log K)。

## 10.5 为什么只训练 assistant 部分？

> 用户 prompt 是条件，不是需要模型学习复现的答案。代码把 prompt 对应 label 设成 `-100`，attention mask 仍然为 1，因此 assistant 可以看到问题，但交叉熵只在回答 token 上计算。

## 10.6 LoRA 和 QLoRA 有什么区别？

> LoRA 冻结 base model，在目标线性层增加低秩 BA 分支，只训练 adapter。QLoRA 进一步把冻结的 base 权重以 4bit NF4 加载，并使用 double quant 和 bf16 计算，从而降低显存。LoRA 参数本身并不是简单用 4bit 训练。

## 10.7 为什么选择 PPO 而不是只做 SFT？

> SFT 只能模仿参考答案，不容易直接优化格式、安全等序列级目标。PPO 可以让模型在线生成，再根据格式、答案覆盖和安全分得到 reward，从而直接优化这些不可微规则指标。同时 reference KL 防止模型偏离 SFT 模型过远。

## 10.8 没有 reward model 还算 PPO 吗？

> 算。PPO 需要 reward signal，但 reward 不一定来自训练过的神经网络。我的主实验使用规则函数产生标量 reward，再通过 TRL-compatible wrapper 接入 PPOTrainer。Policy、reference、value 和 reward 四类功能组件仍然存在。

## 10.9 奖励权重为什么是 0.3/0.5/0.2？

> 第一版根据任务优先级设置：准确性是核心，所以 0.5；格式关系到结构化输出，设为 0.3；安全是底线约束，设为 0.2。这个权重是工程先验，不是理论最优。严谨改进应做单奖励和不同权重的消融，并观察 C-Eval、格式、安全、KL 和长度之间的 Pareto 权衡。

## 10.10 如何保证数据质量？

> 我从规则质量、目标域相关性和实验验证三层控制。规则层做字段、长度、广告、角色和去重；相关性层用 C-Eval 医学题干做 embedding 筛选；验证层通过清洗报告、分数分布、人工抽检和下游 C-Eval/PPL 对比判断。但当前仍缺少系统的医学事实审核，这是项目可以继续补强的地方。

## 10.11 PPL 为什么下降说明有效？

> 在同一份 1K 医疗长回答、相同 tokenizer 和模板下，answer-only PPL 从 12.3325 降到 9.8318，说明 SFT 后模型给参考答案 token 分配了更高概率。但 PPL 只反映语言建模匹配度，不等于医学正确率，所以还要结合 C-Eval 和安全、格式指标。

## 10.12 如何判断 PPO 没把模型训坏？

需要同时看：

- C-Eval 是否明显下降。
- PPL 是否恶化。
- 格式和安全指标是否提升。
- KL 是否失控。
- response length 是否异常。
- 是否出现模板堆砌、过度拒答或关键词作弊。
- 人工 bad case 是否增加。

## 10.13 为什么没有直接把 38 万全部用于主实验？

> 38 万是清洗后的宽领域候选，10 万是与目标医学子域更相关的数据。主实验用 Top100k 是为了验证目标域筛选的价值。更完整的实验应增加 38 万全量 SFT 与随机 10 万对照，从而区分“样本数量”和“筛选策略”的贡献。

## 10.14 你的项目最大不足是什么？

推荐诚实回答：

> 第一，准确率奖励当前主要是关键词覆盖和字符 F1，不是真正的医学事实 verifier；第二，C-Eval target-aware 筛选需要明确污染边界；第三，复杂病例格式指标需要严格区分带提示和无提示；第四，缺少随机 10 万、全量 38 万以及奖励权重消融。这些是我下一步最希望补齐的实验。

---

## 十一、源码定位题

面试官可能直接问“代码在哪里”。至少要能快速定位：

| 问题 | 文件与函数 |
| --- | --- |
| 50 万数据怎么流式读取？ | `scripts/prepare_shibing624_medical_sft.py::iter_source_records()` |
| 文本怎么规范化？ | `scripts/clean_corpus.py::normalize_text()` |
| 多轮角色怎么修复？ | `repair_conversations()` |
| 精确去重在哪里？ | `dedup_key_from_pair()` |
| C-Eval target 怎么读取？ | `scripts/filter_by_ceval_similarity.py::load_targets()` |
| mean pooling 在哪里？ | `TransformersEmbedder.encode()` |
| 余弦相似度在哪里？ | `filter_by_similarity()` 内的矩阵乘法 |
| Top-K 在哪里维护？ | `update_heap()` |
| prompt loss 在哪里 mask？ | `MedicalGPT/training/supervised_finetuning.py::preprocess_function()` |
| QLoRA 4bit 在哪里配置？ | `BitsAndBytesConfig(...)` |
| LoRA target 如何查找？ | `find_all_linear_names()` |
| PPO 数据怎么派生？ | `scripts/build_medical_ppo_dataset.py::convert_row()` |
| 三维奖励在哪里算？ | `compute_total_reward()` |
| reward 如何接入 TRL？ | `RuleBasedRewardModel` |
| value head 在哪里？ | `ValueScoreHead`、`ValueModelWrapper` |
| answer-only PPL 怎么 mask？ | `scripts/evaluate_medical_ppl.py::build_answer_only_features()` |
| PPL 怎么按 token 聚合？ | `evaluate_ppl()` |
| 格式准确率怎么算？ | `scripts/evaluate_complex_case_format.py::compute_format_score()` |

---

## 十二、面试前自测清单

以下问题不看文档能回答，才算真正掌握。

### 数据

- [ ] 能在一分钟内画出 `500000 -> 381621 -> 100000 -> 5000`。
- [ ] 能写出 Alpaca、ShareGPT、PPO 三种 JSON。
- [ ] 能解释 JSONL 流式处理为什么省内存。
- [ ] 能解释 NFKC、HTML、控制字符和零宽字符清理。
- [ ] 能说明为什么按问题加答案做 SHA256 去重。
- [ ] 能解释角色顺序修复。
- [ ] 能说出真实过滤数量。

### 相似筛选

- [ ] 能写出 L2 normalize 和矩阵乘法。
- [ ] 能解释矩阵形状。
- [ ] 能解释最大相似度和最小堆。
- [ ] 能说出 Top-K 复杂度。
- [ ] 能解释 C-Eval 污染边界。
- [ ] 能承认相似度不完全等于质量。

### SFT 与 LoRA

- [ ] 能解释 teacher forcing 和 causal LM loss。
- [ ] 能写出 `labels = [-100] * prompt_len + answer_ids`。
- [ ] 能区分 attention mask 与 label mask。
- [ ] 能解释 Qwen3 template 不等于模型权重。
- [ ] 能写出 LoRA 的 `W + BA`。
- [ ] 能解释 A/B 常见初始化。
- [ ] 能解释 rank、alpha、dropout 和 target modules。
- [ ] 能解释 NF4、double quant 和 bf16 compute。
- [ ] 能算有效 batch size。
- [ ] 能区分 adapter、checkpoint 和 merged model。

### PPO

- [ ] 能画出 policy/reference/value/reward。
- [ ] 能解释为什么规则 reward 仍然可以做 PPO。
- [ ] 能写出三维奖励公式。
- [ ] 能说明准确率分实际是关键词加字符 F1。
- [ ] 能解释 KL 和 PPO clip 的区别。
- [ ] 能解释 value head 的作用。
- [ ] 能说出至少三种 reward hacking。
- [ ] 能解释 PPO 数据和 SFT 数据的关联。

### 评测与工程

- [ ] 能解释 answer-only PPL。
- [ ] 能解释 token 加权聚合。
- [ ] 能区分 prompted 和 unprompted format accuracy。
- [ ] 能解释 C-Eval、PPL、格式指标各自测什么。
- [ ] 能讲清 `.score`、dtype mismatch、left padding 三个 PPO 错误。
- [ ] 能讲一次 OOM 和 checkpoint 排查。
- [ ] 能说明 SwanLab 自动记录什么、自定义指标需要什么。

### 贡献边界

- [ ] 能明确说出哪些代码是自己实现的。
- [ ] 能明确说出哪些模块来自 MedicalGPT。
- [ ] 能明确说出 Transformers、PEFT、TRL 分别做什么。
- [ ] 不把框架调用夸大成底层算法从零实现。
- [ ] 不把规则准确率说成训练 reward model。
- [ ] 不把字符 F1 说成真正 embedding 语义相似度。

---

## 十三、最终复习建议

复习不要按文件顺序死背，按以下顺序更接近真实面试：

```text
第一遍：完整讲出项目链路和个人贡献
第二遍：手写数据 schema、余弦相似度、LoRA 公式、PPO 奖励公式
第三遍：打开源码定位关键函数
第四遍：回答设计选择和局限
第五遍：复盘真实报错和解决方案
```

真正达到面试可用的标准不是“看过源码”，而是：

> 面试官随机指出一个项目模块时，能够先讲清它解决什么问题，再讲输入输出和核心原理，随后定位到真实函数与关键语句，最后说明局限和改进方向。
