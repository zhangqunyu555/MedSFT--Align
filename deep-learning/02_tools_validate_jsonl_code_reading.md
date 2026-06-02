# MedicalGPT 工具源码精读 02：validate_jsonl.py

## 整体作用

`MedicalGPT/tools/validate_jsonl.py` 是一个数据格式校验工具。它检查训练用 JSONL 是否符合 MedicalGPT 的 ShareGPT 对话格式。你现在已经转换好的训练数据：

```text
MedicalGPT/data/sft_medsft_top100k/train.jsonl
MedicalGPT/data/sft_medsft_cleaned_381k/train.jsonl
```

都应该能通过这个工具。

JSONL 的意思是“一行一个 JSON 对象”，不是一个 JSON 数组。MedicalGPT SFT 入口主要读取 `conversations` 字段，所以这个工具重点检查：

- 每行是不是合法 JSON。
- 是否有 `conversations` 字段。
- `conversations` 是否是 list。
- 每条消息是否有 `from` 和 `value`。
- `from` 是否是 `system`、`human`、`gpt`。

## 输入输出

输入：

```text
一个 JSONL 文件路径
```

输出：

```text
终端打印总行数、有效行数、无效行数和错误提示
```

运行命令：

```bash
cd MedicalGPT

python tools/validate_jsonl.py \
  --file_path data/sft_medsft_top100k/train.jsonl
```

## 核心源码：`validate_jsonl(file_path)`

```python
def validate_jsonl(file_path):
    print("开始验证 JSONL 文件格式...\n")

    with open(file_path, 'r', encoding='utf-8') as file:
        line_number = 0
        valid_lines = 0
        total_lines = 0
        for line in file:
            total_lines += 1
            line_number += 1
            try:
                # 尝试解析JSON
                data = json.loads(line)

                # 检查是否包含 'conversations' 键
                if 'conversations' not in data:
                    print(f"第 {line_number} 行: 缺少 'conversations' 键，请检查格式。\n")
                    continue

                # 检查 'conversations' 是否为列表
                conversations = data['conversations']
                if not isinstance(conversations, list):
                    print(f"第 {line_number} 行: 'conversations' 应为列表格式，请检查。\n")
                    continue

                # 检查每个对话是否包含 'from' 和 'value' 键
                conversation_valid = True
                for conv in conversations:
                    if 'from' not in conv or 'value' not in conv:
                        print(f"第 {line_number} 行: 缺少 'from' 或 'value' 键，请检查对话格式。\n")
                        conversation_valid = False
                        continue

                    # 检查 'from' 字段的值是否为 'human' 或 'gpt'
                    if conv['from'] not in ['system', 'human', 'gpt']:
                        print(f"第 {line_number} 行: 'from' 字段的值无效，应为 'human' 或 'gpt'。\n")
                        conversation_valid = False

                if conversation_valid:
                    valid_lines += 1

            except json.JSONDecodeError:
                print(f"第 {line_number} 行: JSON 格式无效，请确保格式正确。\n")

    print(f"验证完成！\n总行数: {total_lines} 行")
    print(f"有效的行数: {valid_lines} 行")
    print(f"无效行数: {total_lines - valid_lines} 行\n")

    if valid_lines == total_lines:
        print("恭喜！所有行的格式都正确。")
    else:
        print("请根据提示检查并修复无效的行。")
```

## 逐段解释

第一段：

```python
with open(file_path, 'r', encoding='utf-8') as file:
```

按 UTF-8 打开 JSONL 文件。中文医疗数据必须显式指定 UTF-8，否则容易出现编码问题。

第二段：

```python
for line in file:
    total_lines += 1
    line_number += 1
```

逐行读取。JSONL 的正确处理方式就是逐行 `json.loads()`，不能对整个文件直接 `json.load()`。

第三段：

```python
data = json.loads(line)
```

把当前行解析成 Python dict。如果这一行不是合法 JSON，就进入 `except json.JSONDecodeError`。

第四段：

```python
if 'conversations' not in data:
```

MedicalGPT 的 SFT 数据核心字段是 `conversations`。如果缺这个字段，训练脚本后面访问 `examples['conversations']` 会失败。

第五段：

```python
conversations = data['conversations']
if not isinstance(conversations, list):
```

`conversations` 必须是 list，因为它表示多轮消息序列。哪怕只有一问一答，也应该是两个 message 组成的 list。

第六段：

```python
for conv in conversations:
    if 'from' not in conv or 'value' not in conv:
```

每条消息必须有：

- `from`：角色
- `value`：消息内容

例如：

```json
{ "from": "human", "value": "患者发热怎么办？" }
```

第七段：

```python
if conv['from'] not in ['system', 'human', 'gpt']:
```

MedicalGPT 接受的普通角色是：

- `system`：系统提示
- `human`：用户
- `gpt`：助手

你的训练数据主要是：

```text
human -> gpt
```

## 和当前 Qwen3 医疗项目的关系

你现在训练 Qwen3-4B-Instruct 前，应该先校验：

```bash
cd MedicalGPT
python tools/validate_jsonl.py --file_path data/sft_medsft_top100k/train.jsonl
```

如果输出：

```text
恭喜！所有行的格式都正确。
```

说明数据格式能进入 SFT 训练入口。

## 常见坑

- 不要用 `python -m json.tool train.jsonl` 校验 JSONL，JSONL 不是一个完整 JSON。
- `from` 写成 `user`、`assistant` 时，这个工具会认为不合法；MedicalGPT 某些训练代码能兼容 `user/assistant`，但工具这里写得更严格。
- `value` 为空不会被这个工具拦住，但训练质量会受影响，所以清洗阶段已经先处理过。
