# 01 数据清洗学习文档

## 1. 本阶段要解决什么问题

本阶段要把约 50 万条中文医疗问答语料整理成后续 SFT 可直接使用的数据。真实医疗语料通常会混有空字段、重复问答、网页标签、广告联系方式、乱码、多轮角色错乱、过短无信息样本、过长拼接样本等问题。如果这些内容直接进入训练，会降低模型医学回答质量，甚至引入危险用药、营销话术和错误医学知识。

本项目的数据清洗脚本是：

```bash
scripts/clean_corpus.py
```

默认配置文件是：

```bash
configs/data_cleaning.yaml
```

默认输入输出路径是：

```text
data/raw/      # 原始 JSONL，不上传 Git
data/cleaned/  # 清洗后 JSONL 和 report，不上传 Git
```

脚本采用流式处理：逐行读取 JSONL，逐条判断，逐条写出。这样可以处理 50 万条甚至更多语料，而不需要一次性把所有数据加载进内存。

## 2. 整体数据流

清洗流程可以理解为一条流水线：

```text
读取配置
  -> 找到输入 JSONL 文件
  -> 逐行读取样本
  -> 解析 JSON
  -> 判断是 ShareGPT 多轮还是单轮问答
  -> 文本标准化
  -> 字段校验
  -> 广告和联系方式过滤
  -> 长度过滤
  -> 精确去重
  -> 写出 Alpaca 或 ShareGPT
  -> 生成 cleaning_report.json
```

代码里的主入口是：

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

意思是：当我们用命令行运行 `python scripts/clean_corpus.py` 时，程序会进入 `main()` 函数，并把 `main()` 的返回值作为进程退出码。

## 3. 运行方式

清洗目录下所有 `.jsonl` 文件：

```bash
python scripts/clean_corpus.py \
  --config configs/data_cleaning.yaml \
  --input data/raw \
  --output data/cleaned
```

清洗单个 `.jsonl` 文件：

```bash
python scripts/clean_corpus.py \
  --config configs/data_cleaning.yaml \
  --input data/raw/medical_qa.jsonl \
  --output data/cleaned
```

输出文件：

```text
data/cleaned/
├── cleaned_alpaca.jsonl
├── cleaned_sharegpt.jsonl
└── cleaning_report.json
```

## 4. 支持的数据格式

### 4.1 单轮问答

输入示例：

```json
{"question":"高血压患者可以运动吗？","answer":"多数高血压患者可以进行中低强度运动，但应结合血压控制情况和医生建议。"}
```

脚本会自动识别常见字段名：

- 问题字段：`question`、`query`、`prompt`、`ask`、`title`
- 回答字段：`answer`、`response`、`output`、`completion`、`reply`

输出为 Alpaca JSONL：

```json
{"instruction":"请回答以下医疗问题","input":"高血压患者可以运动吗?","output":"多数高血压患者可以进行中低强度运动,但应结合血压控制情况和医生建议。"}
```

### 4.2 Alpaca

输入示例：

```json
{"instruction":"请回答以下医疗问题","input":"糖尿病患者饮食要注意什么？","output":"建议控制总热量，规律进餐，减少精制糖摄入，并遵医嘱监测血糖。"}
```

这种格式会继续输出为 Alpaca JSONL。脚本会保留 `instruction`，把 `input` 当作用户问题，把 `output` 当作助手回答。

### 4.3 ShareGPT 多轮对话

输入示例：

```json
{"conversations":[{"from":"user","value":"最近总是胸闷怎么办？"},{"from":"assistant","value":"胸闷原因较多，如持续不缓解或伴胸痛、出汗、呼吸困难，应及时就医。"}]}
```

输出示例：

```json
{"conversations":[{"from":"human","value":"最近总是胸闷怎么办?"},{"from":"gpt","value":"胸闷原因较多,如持续不缓解或伴胸痛、出汗、呼吸困难,应及时就医。"}]}
```

脚本会统一角色：

- 用户角色统一为 `human`
- 助手角色统一为 `gpt`

## 5. 配置文件怎么控制清洗规则

配置文件是 [configs/data_cleaning.yaml](../configs/data_cleaning.yaml)。核心配置如下：

```yaml
input_path: data/raw
output_dir: data/cleaned

min_question_chars: 4
min_answer_chars: 10
max_question_chars: 2048
max_answer_chars: 8192
enable_dedup: true
```

这些参数对应脚本里的默认配置：

```python
DEFAULT_CONFIG: dict[str, Any] = {
    "input_path": "data/raw",
    "output_dir": "data/cleaned",
    "min_question_chars": 4,
    "min_answer_chars": 10,
    "max_question_chars": 2048,
    "max_answer_chars": 8192,
    "enable_dedup": True,
}
```

理解这段代码的关键点：

- `DEFAULT_CONFIG` 是兜底配置。
- 如果配置文件不存在，脚本仍然可以用默认值运行。
- 如果配置文件存在，`load_config()` 会读取配置文件并覆盖默认值。

字段别名配置：

```yaml
question_fields: [question, query, prompt, ask, title]
answer_fields: [answer, response, output, completion, reply]
instruction_fields: [instruction]
input_fields: [input]
output_fields: [output]
conversation_fields: [conversations, messages]
```

如果你的原始数据字段叫 `病人提问`、`医生回答`，当前脚本不会自动识别，需要把字段名加入配置或扩展脚本字段映射。

## 6. 代码执行主线

### 6.1 `main()`：命令行入口

关键代码：

```python
def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.input_path:
        config["input_path"] = args.input_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
```

这段代码做了三件事：

1. `parse_args()` 读取命令行参数。
2. `load_config(args.config)` 读取 YAML 配置。
3. 命令行里的 `--input` 和 `--output` 优先级高于配置文件。

关键判断：

```python
if not input_path.exists():
    print(f"Input path does not exist: {input_path}", file=sys.stderr)
    return 2
```

这句话的意思是：如果输入路径不存在，直接报错退出，避免生成空结果误导后续训练。

### 6.2 `clean_corpus()`：清洗流程总控

关键代码：

```python
def clean_corpus(config: dict[str, Any]) -> Counter:
    input_path = Path(config["input_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
```

这三句负责确定输入输出目录，并自动创建输出目录。

关键代码：

```python
files = iter_jsonl_files(input_path)
report: Counter = Counter()
report["input_files"] = len(files)
```

这里的 `report` 是统计器，后面所有过滤原因都会累加到它里面。例如：

```python
report["invalid_json"] += 1
report["duplicate"] += 1
report["kept"] += 1
```

这就是最后 `cleaning_report.json` 的来源。

最重要的循环是：

```python
for file_path in files:
    with file_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
```

它体现了流式处理思想：外层遍历文件，内层逐行处理。脚本不会把 50 万条数据全部放进列表。

### 6.3 JSON 解析与坏样本过滤

关键代码：

```python
try:
    record = json.loads(line)
except json.JSONDecodeError:
    report["invalid_json"] += 1
    continue
```

如果某一行不是合法 JSON，这一行会被跳过，统计到 `invalid_json`。

关键代码：

```python
if not isinstance(record, dict):
    report["invalid_record"] += 1
    continue
```

本脚本要求每行 JSON 是对象，也就是 `{...}`。如果一行是数组、字符串或数字，就不是可用训练样本。

## 7. 关键函数详解

### 7.1 `parse_args()`：读取命令行参数

函数作用：定义脚本支持哪些命令行参数。

关键代码：

```python
parser.add_argument("--config", default="configs/data_cleaning.yaml")
parser.add_argument("--input", dest="input_path", default=None)
parser.add_argument("--output", dest="output_dir", default=None)
```

含义：

- `--config` 指定配置文件。
- `--input` 指定输入文件或目录。
- `--output` 指定输出目录。

### 7.2 `parse_scalar()`：把配置文本转成 Python 值

函数作用：把配置文件中的字符串转成布尔值、整数、列表或普通字符串。

关键代码：

```python
if value in {"true", "True"}:
    return True
if value in {"false", "False"}:
    return False
```

这让 YAML 里的 `enable_dedup: true` 能变成 Python 的 `True`。

关键代码：

```python
try:
    return int(value)
except ValueError:
    pass
```

这让 `min_question_chars: 4` 变成整数 `4`，后续才能和文本长度比较。

### 7.3 `load_config()`：加载配置并覆盖默认值

函数作用：读取 `configs/data_cleaning.yaml`，把里面的配置覆盖到 `DEFAULT_CONFIG` 上。

关键代码：

```python
config = dict(DEFAULT_CONFIG)
```

这里用 `dict()` 复制默认配置，避免直接修改全局变量。

关键代码：

```python
if not config_path.exists():
    return config
```

如果配置文件不存在，就返回默认配置。这样脚本更稳，不会因为少一个配置文件就直接崩。

关键代码：

```python
config.update(parsed)
return config
```

这表示配置文件优先级高于默认配置。

### 7.4 `iter_jsonl_files()`：找到要处理的文件

函数作用：支持输入单个文件，也支持输入目录。

关键代码：

```python
if input_path.is_file():
    return [input_path]
return sorted(path for path in input_path.rglob("*.jsonl") if path.is_file())
```

含义：

- 如果 `--input` 是文件，就只处理这个文件。
- 如果 `--input` 是目录，就递归查找所有 `.jsonl` 文件。
- `sorted()` 让处理顺序稳定，便于复现。

### 7.5 `normalize_text()`：文本标准化核心函数

函数作用：把不同来源、不同编码风格的文本变成更干净、更一致的文本。

关键代码：

```python
text = html.unescape(text)
```

把网页实体转回来。例如 `&lt;div&gt;` 会变成 `<div>`。

关键代码：

```python
text = unicodedata.normalize("NFKC", text)
```

把全角字母、全角数字等统一为标准形式。例如全角 `Ａ` 会变成半角 `A`。

关键代码：

```python
text = HTML_TAG_RE.sub(" ", text)
text = CONTROL_RE.sub("", text)
```

第一句删除 HTML 标签，第二句删除控制字符。

关键代码：

```python
text = SPACE_RE.sub(" ", text)
text = "\n".join(part.strip() for part in text.splitlines())
text = BLANK_LINE_RE.sub("\n\n", text)
return text.strip()
```

这几句负责压缩空格、清理行首行尾空白、把连续三个以上空行压成两个换行。

为什么这很重要：去重之前必须先标准化，否则“同一条数据因为空格不同”会被当成两条。

### 7.6 `pick_first()`：字段别名选择器

函数作用：从多个可能字段名中选出第一个存在且非空的值。

关键代码：

```python
for field in fields:
    if field in record and record[field] not in (None, ""):
        return record[field]
return ""
```

例子：

```python
pick_first(record, ["question", "query", "prompt"])
```

如果样本里没有 `question`，但有 `query`，就返回 `query` 的值。

### 7.7 `contains_ad_or_contact()`：广告与联系方式过滤

函数作用：判断文本中是否包含广告关键词、URL、手机号、邮箱、微信或 QQ。

关键代码：

```python
if any(keyword and keyword in text for keyword in config["ad_keywords"]):
    return True
```

只要文本包含任何广告关键词，就判定为需要过滤。

关键代码：

```python
return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in config["contact_patterns"])
```

对所有联系方式正则逐个匹配。`re.IGNORECASE` 表示忽略大小写，比如 `VX` 和 `vx` 都能匹配。

### 7.8 `validate_lengths()`：长度过滤

函数作用：判断问题和回答是否为空、过短或过长。

关键代码：

```python
q_len = count_content_chars(question)
a_len = count_content_chars(answer)
```

这里不是简单用 `len(text)`，而是先删除空白再计数，避免“很多空格撑长度”的脏样本通过。

关键判断：

```python
if q_len == 0 or a_len == 0:
    return "empty_field"
if q_len < config["min_question_chars"] or a_len < config["min_answer_chars"]:
    return "too_short"
if q_len > config["max_question_chars"] or a_len > config["max_answer_chars"]:
    return "too_long"
```

返回值不是 `True` / `False`，而是过滤原因。这样后面可以直接写入统计报告：

```python
report[reason] += 1
```

### 7.9 `normalize_role()`：统一多轮角色

函数作用：把不同数据集里的角色名统一成 ShareGPT 训练常用角色。

关键代码：

```python
if role_text in config["user_roles"]:
    return "human"
if role_text in config["assistant_roles"]:
    return "gpt"
return None
```

配置里默认：

```yaml
user_roles: [human, user, 患者, 病人]
assistant_roles: [gpt, assistant, doctor, 医生, 客服]
```

如果角色不在这两个列表里，就返回 `None`，后续会把这条消息视为不可用。

### 7.10 `message_role()` 和 `message_value()`：兼容不同多轮字段名

不同 ShareGPT 数据可能用不同字段：

```json
{"from":"user","value":"问题"}
```

或：

```json
{"role":"user","content":"问题"}
```

脚本用两个函数兼容：

```python
def message_role(message: dict[str, Any]) -> Any:
    return message.get("from", message.get("role", ""))

def message_value(message: dict[str, Any]) -> Any:
    return message.get("value", message.get("content", ""))
```

含义是：优先取 `from` / `value`，如果没有就取 `role` / `content`。

### 7.11 `repair_conversations()`：多轮对话修复核心函数

这是多轮数据最关键的函数。

第一步：必须是列表。

```python
if not isinstance(conversations, list):
    return None, False
```

第二步：逐条消息清洗角色和值。

```python
role = normalize_role(message_role(message), config)
value = normalize_text(message_value(message))
if role is None or not value:
    changed = True
    continue
```

如果角色不认识，或消息内容为空，就丢弃这条消息。

第三步：合并连续同角色消息。

```python
if repaired and repaired[-1]["from"] == role:
    repaired[-1]["value"] = normalize_text(repaired[-1]["value"] + "\n" + value)
    changed = True
```

例如原始数据是：

```text
user: 最近胸闷怎么办？
user: 还会出汗
assistant: 建议及时就医
```

会被修成：

```text
human: 最近胸闷怎么办？
       还会出汗
gpt: 建议及时就医
```

第四步：保证第一条是用户。

```python
while repaired and repaired[0]["from"] != "human":
    repaired.pop(0)
    changed = True
```

如果数据开头是助手欢迎语，例如 `assistant: 您好`，就删除它。

第五步：保证最后一条是助手。

```python
while repaired and repaired[-1]["from"] != "gpt":
    repaired.pop()
    changed = True
```

如果最后只有用户追问但没有回答，就删除尾部用户消息。

第六步：校验角色必须严格交替。

```python
for index, message in enumerate(repaired):
    expected = "human" if index % 2 == 0 else "gpt"
    if message["from"] != expected:
        return None, True
```

这句话很关键：偶数位置必须是 `human`，奇数位置必须是 `gpt`。不满足就过滤整条样本。

### 7.12 `extract_single_turn()`：抽取单轮样本

函数作用：把 Alpaca 或普通 QA 样本抽成统一三元组：

```text
instruction, input_text, answer
```

关键代码：

```python
instruction = normalize_text(pick_first(record, config["instruction_fields"]))
input_text = normalize_text(pick_first(record, config["input_fields"]))
output = normalize_text(pick_first(record, config["output_fields"]))

if instruction and output:
    return instruction, input_text, output
```

如果样本有 `instruction` 和 `output`，优先按 Alpaca 格式处理。

否则按普通 QA 处理：

```python
question = normalize_text(pick_first(record, config["question_fields"]))
answer = normalize_text(pick_first(record, config["answer_fields"]))
return "请回答以下医疗问题", question, answer
```

这里给普通 QA 自动补了一个统一 instruction：`请回答以下医疗问题`。

### 7.13 `dedup_key_from_pair()`：单轮去重 key

函数作用：对单轮问题和回答计算去重 key。

关键代码：

```python
text = compact_for_dedup(question) + "\n" + compact_for_dedup(answer)
return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

`compact_for_dedup()` 会去掉空白并转小写：

```python
return re.sub(r"\s+", "", text.lower())
```

所以这两条会被认为是重复：

```text
高血压 可以 运动吗？
高血压可以运动吗?
```

### 7.14 `dedup_key_from_messages()`：多轮去重 key

函数作用：对多轮对话的全部角色和值一起计算 hash。

关键代码：

```python
text = "\n".join(f"{item['from']}:{compact_for_dedup(item['value'])}" for item in messages)
return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

这里把角色也放进 hash，避免以下两类样本被误判为重复：

```text
human: A
gpt: B
```

和：

```text
gpt: A
human: B
```

### 7.15 `write_jsonl()`：写出一行 JSON

关键代码：

```python
handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
```

重要点：

- `ensure_ascii=False`：中文正常写出，不转成 `\u4e2d\u6587`。
- `separators=(",", ":")`：减少多余空格，让 JSONL 更紧凑。
- 每条样本后加 `\n`，保证一行一条 JSON。

## 8. `clean_corpus()` 中最关键的分支

### 8.1 先判断多轮样本

关键代码：

```python
conversations = extract_conversations(record, config)
if conversations:
    messages, changed = repair_conversations(conversations, config)
```

只要样本里有 `conversations` 或 `messages`，就优先按多轮处理。

多轮失败时：

```python
if not messages:
    report["multi_turn_failed"] += 1
    continue
```

`continue` 的意思是：这条样本到此结束，不再进入后续写出逻辑。

多轮通过后还要过滤广告和联系方式：

```python
text_for_filter = "\n".join(item["value"] for item in messages)
if contains_ad_or_contact(text_for_filter, config):
    report["ad_or_contact"] += 1
    continue
```

多轮去重：

```python
key = dedup_key_from_messages(messages)
if config["enable_dedup"] and key in seen:
    report["duplicate"] += 1
    continue
seen.add(key)
```

最终写出：

```python
write_jsonl(sharegpt_out, {"conversations": messages})
report["sharegpt_kept"] += 1
report["kept"] += 1
```

### 8.2 如果不是多轮，就按单轮样本处理

关键代码：

```python
instruction, input_text, answer = extract_single_turn(record, config)
```

随后做长度过滤：

```python
reason = validate_lengths(question_for_validation, answer, config)
if reason:
    report[reason] += 1
    continue
```

这句话是单轮过滤的核心：只要 `validate_lengths()` 返回了原因，就统计并跳过。

广告过滤：

```python
if contains_ad_or_contact(question + "\n" + answer, config):
    report["ad_or_contact"] += 1
    continue
```

去重：

```python
key = dedup_key_from_pair(question_for_validation, answer)
if config["enable_dedup"] and key in seen:
    report["duplicate"] += 1
    continue
seen.add(key)
```

写出 Alpaca：

```python
write_jsonl(
    alpaca_out,
    {"instruction": instruction, "input": input_text, "output": answer},
)
report["alpaca_kept"] += 1
report["kept"] += 1
```

## 9. 清洗报告怎么读

`cleaning_report.json` 示例：

```json
{
  "input_path": "data/raw",
  "output_dir": "data/cleaned",
  "alpaca_output": "data/cleaned/cleaned_alpaca.jsonl",
  "sharegpt_output": "data/cleaned/cleaned_sharegpt.jsonl",
  "stats": {
    "total_read": 500000,
    "kept": 380000,
    "alpaca_kept": 350000,
    "sharegpt_kept": 30000,
    "duplicate": 60000,
    "too_short": 25000,
    "too_long": 8000,
    "ad_or_contact": 12000,
    "invalid_json": 300
  }
}
```

关键字段：

| 字段 | 含义 | 如果数值很高说明什么 |
| --- | --- | --- |
| `total_read` | 总读取样本数 | 应接近原始数据量 |
| `kept` | 成功保留样本数 | 如果太低，可能规则过严 |
| `alpaca_kept` | 单轮输出数 | 用于 SFT 主训练集 |
| `sharegpt_kept` | 多轮输出数 | 用于多轮对话 SFT |
| `empty_field` | 空字段过滤数 | 字段名可能没配对 |
| `too_short` | 过短过滤数 | 数据可能有大量无意义短文本 |
| `too_long` | 过长过滤数 | 可能有网页拼接或爬虫污染 |
| `ad_or_contact` | 广告联系方式过滤数 | 可能有营销数据或规则过严 |
| `duplicate` | 重复过滤数 | 数据源重复率高 |
| `invalid_json` | 非法 JSON 行数 | 原始文件格式可能损坏 |
| `multi_turn_repaired` | 多轮修复次数 | 多轮角色格式不统一 |
| `multi_turn_failed` | 多轮失败数 | 角色或内容字段可能不兼容 |

## 10. 如何修改规则

### 10.1 调整长度

如果你发现回答稍长但质量很好，可以调大：

```yaml
max_answer_chars: 12000
```

如果短回答太多，可以调大：

```yaml
min_answer_chars: 20
```

### 10.2 增加字段别名

如果原始数据是：

```json
{"病人提问":"头痛怎么办？","医生回答":"如果持续头痛或伴恶心呕吐，应及时就医。"}
```

可以把配置改成：

```yaml
question_fields: [question, query, prompt, ask, title, 病人提问]
answer_fields: [answer, response, output, completion, reply, 医生回答]
```

### 10.3 放宽广告过滤

如果手机号并不是广告，而是医学文本里的示例，可以从 `contact_patterns` 中删除手机号正则：

```yaml
- 1[3-9]\d{9}
```

但医疗问答训练通常不建议保留真实联系方式。

### 10.4 关闭去重

调试时可以临时关闭：

```yaml
enable_dedup: false
```

正式训练数据建议开启去重。

## 11. 小样例自测

可以用临时目录做测试，不要把真实数据提交到 Git：

```bash
mkdir -p /private/tmp/medsft-clean-test/input
mkdir -p /private/tmp/medsft-clean-test/output
```

写入一个测试文件 `/private/tmp/medsft-clean-test/input/sample.jsonl`：

```jsonl
{"question":"高血压可以运动吗？","answer":"多数高血压患者可以进行中低强度运动，但应结合血压控制情况并遵医嘱。"}
{"question":"高血压可以运动吗？","answer":"多数高血压患者可以进行中低强度运动，但应结合血压控制情况并遵医嘱。"}
{"question":"糖尿病饮食？","answer":"加微信免费咨询 13812345678"}
{"instruction":"请回答以下医疗问题","input":"感冒发烧需要马上吃抗生素吗？","output":"普通感冒多由病毒引起，是否使用抗生素需要医生根据感染证据判断，不建议自行用药。"}
{"conversations":[{"from":"assistant","value":"您好"},{"from":"user","value":"最近胸闷怎么办？"},{"from":"user","value":"还会出汗"},{"from":"assistant","value":"胸闷伴出汗需要警惕心血管问题，如持续不缓解应及时就医。"}]}
{"question":"短","answer":"太短"}
```

运行：

```bash
python scripts/clean_corpus.py \
  --config configs/data_cleaning.yaml \
  --input /private/tmp/medsft-clean-test/input \
  --output /private/tmp/medsft-clean-test/output
```

预期现象：

- 第一条保留。
- 第二条因为重复被过滤。
- 第三条因为广告和手机号被过滤。
- 第四条按 Alpaca 保留。
- 第五条修复为 ShareGPT，前置助手欢迎语会删除，连续用户消息会合并。
- 第六条因为过短被过滤。

## 12. 验收标准

本阶段完成后应满足：

- 能流式读取目录下多个 JSONL 文件。
- 能输出 Alpaca 和 ShareGPT 两类清洗结果。
- 能生成 `cleaning_report.json`。
- 重复样本、广告样本、空字段样本、过短过长样本会被过滤。
- 多轮对话能统一角色并修复常见顺序问题。
- `data/raw/` 和 `data/cleaned/` 中的真实数据不会被 Git 提交。

建议验证命令：

```bash
python3 -m py_compile scripts/clean_corpus.py
git check-ignore data/raw/sample.jsonl data/cleaned/cleaned_alpaca.jsonl
git check-ignore README.md REPRODUCTION_PLAN.md configs/data_cleaning.yaml docs/01_data_cleaning.md scripts/clean_corpus.py
```

最后一条命令预期没有输出，因为这些源码、配置和文档不应该被忽略。

## 13. 常见问题

### 13.1 运行时报 `Input path does not exist`

说明 `--input` 指定的路径不存在。先创建目录：

```bash
mkdir -p data/raw
```

然后把原始 `.jsonl` 文件放进去。

### 13.2 输出为空

优先查看：

```bash
data/cleaned/cleaning_report.json
```

常见原因：

- 字段名不在 `question_fields` / `answer_fields` 中。
- 回答太短，被 `min_answer_chars` 过滤。
- 数据中包含大量联系方式，被 `contact_patterns` 过滤。
- 多轮数据字段不是 `conversations` / `messages`。

### 13.3 ShareGPT 样本大量失败

检查原始数据中的角色字段是否是 `from` / `role`，内容字段是否是 `value` / `content`。如果字段不同，需要扩展：

```python
message_role()
message_value()
```

### 13.4 重复过滤太多

说明多个数据源中存在大量相同问答，或标准化后文本非常接近。当前脚本只做精确去重，不做语义近似去重；如果重复过滤数很高，通常是正常现象。

### 13.5 为什么当前不输出 Preference JSONL

本阶段目标是 SFT 语料清洗。Preference 数据需要 `chosen` / `rejected`，通常来自强模型生成、人工标注或规则构造。后续 DPO / PPO 阶段会单独实现 preference 数据构造脚本。

