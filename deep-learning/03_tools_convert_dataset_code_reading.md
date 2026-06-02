# MedicalGPT 工具源码精读 03：convert_dataset.py

## 整体作用

`MedicalGPT/tools/convert_dataset.py` 用来把不同格式的数据转换成 MedicalGPT 训练常用的 ShareGPT JSONL。

它支持：

- Alpaca -> ShareGPT
- QA -> ShareGPT
- JSON array -> JSONL
- 已经是 ShareGPT 的数据，只保留 `conversations`

你的项目里最相关的是 Alpaca -> ShareGPT：

```text
{instruction,input,output}
  -> {conversations:[{from,value},{from,value}]}
```

## 输入输出

输入 Alpaca：

```json
{"instruction": "请回答以下医疗问题", "input": "发热怎么办？", "output": "应结合体温、症状判断..."}
```

输出 ShareGPT：

```json
{"conversations":[{"from":"human","value":"请回答以下医疗问题\n\n发热怎么办？"},{"from":"gpt","value":"应结合体温、症状判断..."}]}
```

运行命令：

```bash
cd MedicalGPT

python tools/convert_dataset.py \
  --in_file ../data/sft/shibing624_medical_top100k.jsonl \
  --out_file data/sft_medsft_top100k/train.jsonl \
  --data_type alpaca \
  --file_type jsonl
```

## 源码分块一：参数解析

```python
parser = argparse.ArgumentParser()
parser.add_argument("--in_file", type=str, required=True, help="Input file name.")
parser.add_argument("--out_file", type=str, required=True, help="Output file name, e.g. out.jsonl")
parser.add_argument("--data_type", type=str, default='alpaca',
                    help="alpaca, qa, json2jsonl, or sharegpt")
parser.add_argument("--file_type", type=str, default='json', help='Input file type: json or csv')
args = parser.parse_args()
print(args)
```

## 逐段解释

- `--in_file`：输入文件。
- `--out_file`：输出 JSONL。
- `--data_type`：告诉脚本输入数据属于哪种逻辑格式。
- `--file_type`：告诉脚本物理文件是 JSON、JSONL 还是 CSV。

这两个概念不同：

```text
data_type = 数据字段结构
file_type = 文件存储格式
```

## 源码分块二：JSON array 转 JSONL

```python
if args.data_type == 'json2jsonl':
    with open(args.in_file) as f:
        data = json.load(f)
    with open(args.out_file, 'w') as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(f"Converted {len(data)} samples: {args.in_file} -> {args.out_file}")
```

## 逐段解释

这段只处理一种情况：输入是 JSON 数组。

```json
[
  {"a": 1},
  {"a": 2}
]
```

输出是 JSONL：

```jsonl
{"a": 1}
{"a": 2}
```

`ensure_ascii=False` 用来保留中文。

## 源码分块三：读取 CSV / JSON / JSONL

```python
data_files = {"train": args.in_file}
if args.file_type == 'csv':
    if args.data_type == 'qa':
        column_names = ['input', 'output']
    else:
        column_names = ['instruction', 'input', 'output']
    raw_datasets = load_dataset('csv', data_files=data_files, column_names=column_names, delimiter='\t')
elif args.file_type in ['json', 'jsonl']:
    raw_datasets = load_dataset('json', data_files=data_files)
else:
    raise ValueError(f"File type not supported: {args.file_type}")
ds = raw_datasets['train']
```

## 逐段解释

- `load_dataset('csv', ...)`：用 Hugging Face datasets 读取 CSV/TSV。
- `delimiter='\t'`：这里按制表符切分，所以更准确说是 TSV。
- `load_dataset('json', ...)`：datasets 同时支持 JSON 和 JSONL。
- `raw_datasets['train']`：把输入文件当成 train split。

## 源码分块四：`process_qa()`

```python
def process_qa(examples):
    convs = []
    for q, a in zip(examples['input'], examples['output']):
        convs.append([
            {"from": "human", "value": q},
            {"from": "gpt", "value": a}
        ])
    return {"conversations": convs}
```

## 逐段解释

QA 格式只有：

```text
input
output
```

所以用户问题直接用 `input`，助手回答直接用 `output`。

输出结构是：

```text
human -> gpt
```

## 源码分块五：`process_alpaca()`

```python
def process_alpaca(examples):
    convs = []
    for instruction, inp, output in zip(examples['instruction'], examples['input'], examples['output']):
        if inp and len(inp.strip()) > 0:
            instruction = instruction + '\n\n' + inp
        convs.append([
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": output}
        ])
    return {"conversations": convs}
```

## 逐段解释

- 遍历一批 Alpaca 样本。
- 如果 `input` 非空，就把 `instruction` 和 `input` 拼起来。
- 两个换行 `\n\n` 用来让任务指令和具体问题分开。
- `output` 作为 `gpt` 的回答。

这正是你当前 Qwen3 医疗 SFT 数据的转换逻辑。

## 源码分块六：执行转换并写出

```python
if args.data_type == 'alpaca':
    ds = ds.map(process_alpaca, batched=True, remove_columns=ds.column_names, desc="Running process")
elif args.data_type == 'qa':
    ds = ds.map(process_qa, batched=True, remove_columns=ds.column_names, desc="Running process")
else:
    if "items" in ds.column_names:
        ds = ds.rename(columns={"items": "conversations"})
    columns_to_remove = ds.column_names.copy()
    columns_to_remove.remove('conversations')
    ds = ds.remove_columns(columns_to_remove)

ds.to_json(f"{args.out_file}", lines=True, force_ascii=False)
```

## 逐段解释

- `batched=True`：批量处理，速度更快。
- `remove_columns=ds.column_names`：删除原字段，只保留返回的新字段。
- 如果不是 Alpaca / QA，就假设数据已经是 ShareGPT，只保留 `conversations`。
- `to_json(..., lines=True)`：输出 JSONL。

## 和当前项目的关系

我们后来自己写了：

```text
scripts/convert_alpaca_to_sharegpt.py
```

原因是本机没有安装 `datasets`，而 MedicalGPT 这个工具依赖：

```python
from datasets import load_dataset
```

功能上两者做的事相同：Alpaca -> ShareGPT。区别是：

- MedicalGPT 版本依赖 datasets，适合完整训练环境。
- 我们的版本无第三方依赖，适合快速转换大 JSONL。

## 常见坑

- CSV 分隔符是 `\t`，不是逗号。
- Alpaca 数据必须有 `instruction/input/output`。
- 输出的是 ShareGPT，不是 Qwen3 特殊 token。Qwen3 模板在训练阶段由 `--template_name qwen3` 处理。

