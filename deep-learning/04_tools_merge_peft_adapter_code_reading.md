# MedicalGPT 工具源码精读 04：merge_peft_adapter.py

## 整体作用

`merge_peft_adapter.py` 用来把 LoRA / PEFT adapter 合并到底座模型里，导出一个完整 Hugging Face 模型目录。

训练 LoRA 后通常有两种使用方式：

1. 不合并：推理或评估时同时加载 base model + adapter。
2. 合并：把 adapter 权重融合进 base model，保存成完整模型。

当前你用 lm-evaluation-harness 评估时可以直接用：

```text
pretrained=Qwen3-4B-Instruct,peft=outputs/qwen3_4b_medical_qlora_top100k
```

所以不一定要合并。但如果你要部署、量化、上传完整模型，就可以用这个工具。

## 运行命令

```bash
cd MedicalGPT

python tools/merge_peft_adapter.py \
  --base_model /workspace/models/Qwen3-4B-Instruct \
  --lora_model outputs/qwen3_4b_medical_qlora_top100k \
  --output_dir outputs/qwen3_4b_medical_merged
```

## 源码分块一：参数解析

```python
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, required=True, type=str,
                    help="Base model name or path")
parser.add_argument('--tokenizer_path', default=None, type=str,
                    help="Please specify tokenization path.")
parser.add_argument('--lora_model', default=None, required=True, type=str,
                    help="Please specify LoRA model to be merged.")
parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
parser.add_argument('--output_dir', default='./merged', type=str)
parser.add_argument('--hf_hub_model_id', default='', type=str)
parser.add_argument('--hf_hub_token', default=None, type=str)
args = parser.parse_args()
```

## 逐段解释

- `--base_model`：底座模型路径，例如 Qwen3-4B-Instruct。
- `--lora_model`：训练得到的 LoRA adapter 目录。
- `--tokenizer_path`：可选 tokenizer 路径，不传就用 base model 的 tokenizer。
- `--resize_emb`：如果 tokenizer 词表变了，需要扩展 embedding。
- `--output_dir`：合并后完整模型保存目录。
- `--hf_hub_model_id`：可选，合并后直接推到 Hugging Face Hub。

## 源码分块二：读取 PEFT 配置并判断任务类型

```python
peft_config = PeftConfig.from_pretrained(lora_model_path)

if peft_config.task_type == "SEQ_CLS":
    print("Loading LoRA for sequence classification model")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=1,
        load_in_8bit=False,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
else:
    print("Loading LoRA for causal language model")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype='auto',
        trust_remote_code=True,
        device_map="auto",
    )
```

## 逐段解释

- `PeftConfig.from_pretrained(lora_model_path)` 读取 adapter 目录里的 `adapter_config.json`。
- 如果任务是 `SEQ_CLS`，说明是奖励模型或分类模型。
- 否则默认是 causal language model，也就是 SFT 后的聊天模型。
- `device_map="auto"` 让 transformers 自动把模型放到可用 GPU。

你的 Qwen3 医疗 SFT adapter 属于 causal LM，所以走 `AutoModelForCausalLM`。

## 源码分块三：加载 tokenizer 和 resize embedding

```python
if args.tokenizer_path:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if args.resize_emb:
    base_model_token_size = base_model.get_input_embeddings().weight.size(0)
    if base_model_token_size != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"Resize vocabulary size {base_model_token_size} to {len(tokenizer)}")
```

## 逐段解释

- tokenizer 默认从 base model 加载。
- 如果你训练时扩展过词表，就需要 `--resize_emb`。
- 当前 Qwen3 医疗 SFT 没有改 tokenizer，通常不需要 `--resize_emb`。

## 源码分块四：加载 adapter 并合并

```python
new_model = PeftModel.from_pretrained(
    base_model,
    lora_model_path,
    device_map="auto",
    torch_dtype='auto',
)
new_model.eval()
print(f"Merging with merge_and_unload...")
base_model = new_model.merge_and_unload()
```

## 逐段解释

- `PeftModel.from_pretrained(base_model, lora_model_path)` 把 LoRA adapter 挂到底座模型上。
- `new_model.eval()` 切换到推理模式。
- `merge_and_unload()` 是核心：把 LoRA 权重合并进原模型权重，并卸载 LoRA 结构。

合并前：

```text
base model + adapter
```

合并后：

```text
merged full model
```

## 源码分块五：保存和上传

```python
tokenizer.save_pretrained(output_dir)
base_model.save_pretrained(output_dir, max_shard_size='10GB')
print(f"Done! model saved to {output_dir}")
if args.hf_hub_model_id:
    print(f"Pushing to Hugging Face Hub...")
    base_model.push_to_hub(
        args.hf_hub_model_id,
        token=args.hf_hub_token,
        max_shard_size="10GB",
    )
    tokenizer.push_to_hub(
        args.hf_hub_model_id,
        token=args.hf_hub_token,
    )
```

## 逐段解释

- 保存 tokenizer。
- 保存合并后的模型。
- `max_shard_size='10GB'` 表示大模型权重会按 10GB 分片。
- 如果传了 Hub ID，就上传模型。

## 和当前项目的关系

评估阶段不一定合并，因为 harness 支持 `peft=adapter_path`。部署阶段可以合并，尤其是：

- 想用 vLLM 加载完整模型。
- 想做离线量化。
- 想上传一个完整模型目录。

## 常见坑

- 合并需要足够显存或内存。
- QLoRA adapter 合并时仍然要加载 base model。
- 合并后模型很大，不要上传 Git。

