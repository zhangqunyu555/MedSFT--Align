# MedicalGPT SFT 训练操作说明

本阶段目标是把当前项目已经准备好的中文医疗 SFT 数据，交给本地 `MedicalGPT/` 子项目完成 `Qwen3-4B-Instruct` 的 LoRA / QLoRA 指令微调。`MedicalGPT/` 是外部参考训练框架，已经加入根仓库 `.gitignore`，所以它不会被提交到当前 `MedSFT-Align` 仓库。

## 1. 当前数据怎么选

当前项目里有三份关键数据：

| 文件 | 条数 | 用途 |
| --- | ---: | --- |
| `data/raw/shibing624_medical/medical_zh_500k.jsonl` | 500000 | 从 `shibing624/medical` 取出的 50 万候选集 |
| `data/cleaned/shibing624_medical_500k/cleaned_alpaca.jsonl` | 381621 | 经过规则清洗后的 Alpaca 候选集 |
| `data/sft/shibing624_medical_top100k.jsonl` | 100000 | 按 C-Eval 医学目标集向量相似度筛出的 10 万高相似 SFT 数据 |

主实验建议使用：

```bash
data/sft/shibing624_medical_top100k.jsonl
```

原因是这 10 万条已经经过两层处理：先从 50 万候选集中清洗，再用 C-Eval 临床医学和基础医学目标集筛选，和本项目的“目标域医学能力提升”最一致。

`cleaned_alpaca.jsonl` 可以作为对照实验使用，用来比较“全部清洗数据训练”和“相似度筛选数据训练”的差异。

## 2. MedicalGPT 需要什么格式

我们当前的 10 万数据是 Alpaca JSONL，每行大致是：

```json
{
  "instruction": "请回答以下医疗问题",
  "input": "患者出现胸痛、气短，可能是什么原因？",
  "output": "胸痛和气短可能与心血管、呼吸系统等多种疾病有关..."
}
```

`MedicalGPT` 的 SFT 训练入口实际更偏向 ShareGPT conversations 格式：

```json
{
  "conversations": [
    {"from": "human", "value": "请回答以下医疗问题\n\n患者出现胸痛、气短，可能是什么原因？"},
    {"from": "gpt", "value": "胸痛和气短可能与心血管、呼吸系统等多种疾病有关..."}
  ]
}
```

`MedicalGPT/tools/convert_dataset.py` 已经提供了 Alpaca 到 ShareGPT 的转换逻辑，关键函数是 `process_alpaca()`：

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

这段逻辑的意思是：

- `instruction` 和 `input` 会拼成用户问题。
- `output` 会作为模型要学习的助手回答。
- 最终训练时只对助手回答部分计算 loss，用户问题部分会被 mask 掉。

## 3. 准备训练数据

先进入 MedicalGPT 子项目：

```bash
cd MedicalGPT
```

建议单独建一个训练数据目录：

```bash
mkdir -p data/sft_medsft_top100k
```

把根项目的 10 万 Alpaca 数据转换成 MedicalGPT 使用的 ShareGPT 格式：

```bash
cd ..
python scripts/convert_alpaca_to_sharegpt.py \
  --input data/sft/shibing624_medical_top100k.jsonl \
  --output MedicalGPT/data/sft_medsft_top100k/train.jsonl \
  --log-every 10000
cd MedicalGPT
```

转换后检查行数：

```bash
wc -l data/sft_medsft_top100k/train.jsonl
```

期望输出是：

```text
100000 data/sft_medsft_top100k/train.jsonl
```

如果要跑对照实验，可以把清洗后的 381621 条也转换一份：

```bash
cd ..

python scripts/convert_alpaca_to_sharegpt.py \
  --input data/cleaned/shibing624_medical_500k/cleaned_alpaca.jsonl \
  --output MedicalGPT/data/sft_medsft_cleaned_381k/train.jsonl \
  --log-every 10000

cd MedicalGPT
```

## 4. 安装训练环境

在 `MedicalGPT/` 目录下安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 QLoRA，需要 CUDA 环境和 bitsandbytes。Mac 的 MPS 通常不适合完整跑 `Qwen3-4B-Instruct` 的 QLoRA，建议只做小样本 smoke test，正式训练放到 NVIDIA CUDA 机器上。

推荐环境：

| 场景 | 建议 |
| --- | --- |
| Mac / MPS | 跑数据格式验证、小样本 LoRA smoke test |
| 单卡 24GB CUDA | 尝试 QLoRA，batch size 小一点 |
| A100 / H100 或多卡 | 正式 10 万数据训练 |

## 5. 先跑 smoke test

第一次不要直接跑 10 万条，先用 200 条确认数据、模板、显存都没问题：

```bash
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --train_file_dir data/sft_medsft_top100k \
  --validation_file_dir data/sft_medsft_top100k \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples 200 \
  --max_eval_samples 20 \
  --model_max_length 1024 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --warmup_steps 10 \
  --logging_steps 10 \
  --eval_steps 50 \
  --save_steps 100 \
  --save_total_limit 2 \
  --preprocessing_num_workers 4 \
  --output_dir outputs/qwen3_4b_medical_lora_smoke \
  --template_name qwen3 \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --gradient_checkpointing True \
  --report_to tensorboard
```

smoke test 通过的标准：

- 能正常加载 tokenizer 和模型。
- 能找到 `data/sft_medsft_top100k/train.jsonl`。
- 日志中能看到 train/eval dataset。
- 开始训练后 loss 能正常打印。
- `outputs/qwen3_4b_medical_lora_smoke/` 下生成 LoRA adapter 文件。

## 6. 正式 LoRA SFT 命令

如果不用 4bit 量化，只做普通 LoRA，可以用：

```bash
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --train_file_dir data/sft_medsft_top100k \
  --validation_file_dir data/sft_medsft_top100k \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples -1 \
  --max_eval_samples 1000 \
  --model_max_length 2048 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.05 \
  --logging_strategy steps \
  --logging_steps 20 \
  --eval_strategy steps \
  --eval_steps 500 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 3 \
  --preprocessing_num_workers 4 \
  --output_dir outputs/qwen3_4b_medical_lora_top100k \
  --template_name qwen3 \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --gradient_checkpointing True \
  --report_to tensorboard
```

这里几个关键参数的作用：

| 参数 | 作用 |
| --- | --- |
| `--train_file_dir` | 指向训练 JSONL 所在目录，MedicalGPT 会递归读取目录下所有 `.jsonl` |
| `--validation_file_dir` | 指向验证 JSONL 所在目录；没有单独验证集时可先复用训练目录，并限制 `--max_eval_samples` |
| `--template_name qwen3` | 使用 MedicalGPT 内置 Qwen3 对话模板 |
| `--use_peft True` | 启用 LoRA / PEFT，只训练少量 adapter 参数 |
| `--target_modules q_proj,k_proj,v_proj,o_proj` | 对 Qwen attention 投影层注入 LoRA |
| `--model_max_length 2048` | 单条样本最大 token 长度，显存紧张可改成 1024 |
| `--gradient_accumulation_steps 16` | 用梯度累积模拟更大的 batch |

## 7. QLoRA 命令

如果是 NVIDIA CUDA 环境，并且安装好了 bitsandbytes，可以启用 QLoRA：

```bash
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --train_file_dir data/sft_medsft_top100k \
  --validation_file_dir data/sft_medsft_top100k \
  --do_train \
  --do_eval \
  --use_peft True \
  --qlora True \
  --load_in_4bit True \
  --max_train_samples -1 \
  --max_eval_samples 1000 \
  --model_max_length 2048 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.05 \
  --logging_steps 20 \
  --eval_steps 500 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --preprocessing_num_workers 4 \
  --output_dir outputs/qwen3_4b_medical_qlora_top100k \
  --template_name qwen3 \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --optim paged_adamw_32bit \
  --gradient_checkpointing True \
  --report_to tensorboard
```

`--qlora True --load_in_4bit True` 是 QLoRA 的核心组合。它会把底座模型用 4bit 方式加载，只训练 LoRA adapter，从而降低显存占用。

如果机器不支持 `bf16`，把下面两项去掉或改成 `float16`：

```bash
--torch_dtype bfloat16
--bf16
```

## 8. 多卡训练

多卡时使用 `torchrun`：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training/supervised_finetuning.py \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --train_file_dir data/sft_medsft_top100k \
  --validation_file_dir data/sft_medsft_top100k \
  --do_train \
  --do_eval \
  --use_peft True \
  --qlora True \
  --load_in_4bit True \
  --max_train_samples -1 \
  --max_eval_samples 1000 \
  --model_max_length 2048 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --logging_steps 20 \
  --eval_steps 500 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --output_dir outputs/qwen3_4b_medical_qlora_top100k \
  --template_name qwen3 \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --optim paged_adamw_32bit \
  --gradient_checkpointing True \
  --ddp_find_unused_parameters False \
  --ddp_timeout 30000 \
  --report_to tensorboard
```

## 9. 训练过程怎么看

训练日志重点看：

- `loss`：应该逐步下降或至少稳定波动。
- `eval_loss`：如果突然变成 `nan`，优先降低学习率或缩短 `model_max_length`。
- 显存占用：如果 OOM，优先把 `model_max_length` 从 2048 降到 1024，再把 batch size 保持为 1。
- checkpoint：确认 `outputs/.../checkpoint-*` 正常生成。

查看 TensorBoard：

```bash
tensorboard --logdir outputs/qwen3_4b_medical_qlora_top100k
```

## 10. 训练后的产物

LoRA / QLoRA 默认保存的是 adapter，不是完整合并模型。常见文件包括：

```text
adapter_config.json
adapter_model.safetensors
training_args.bin
tokenizer.json
tokenizer_config.json
```

后续推理时需要同时加载：

- base model：`Qwen/Qwen3-4B-Instruct`
- lora adapter：`outputs/qwen3_4b_medical_qlora_top100k`

如果要导出合并后的完整模型，再使用 MedicalGPT 提供的 adapter merge 工具；合并后的模型文件通常很大，仍然不应该提交到 Git。

## 11. 常见问题

### 11.1 为什么不直接把 Alpaca JSONL 丢给训练脚本

`training/supervised_finetuning.py` 的预处理核心读取的是 `examples['conversations']`。虽然我们自己的根项目保留 Alpaca 格式更方便做清洗和筛选，但交给 MedicalGPT 训练前，建议先转成 ShareGPT conversations 格式，减少字段不匹配风险。

### 11.2 `template_name` 为什么用 `qwen3`

MedicalGPT 的 `training/template.py` 注册了 `qwen3` 和 `qwen3_nothink` 模板。对 `Qwen3-4B-Instruct`，主实验用 `qwen3`；如果希望模型少输出思考过程，可以对照尝试 `qwen3_nothink`。

### 11.3 `target_modules` 用 `q_proj,k_proj,v_proj,o_proj` 还是 `all`

普通 LoRA 建议先用：

```text
q_proj,k_proj,v_proj,o_proj
```

它更稳，训练参数更少。QLoRA 如果显存允许，可以用：

```text
all
```

MedicalGPT 会自动寻找线性层，覆盖范围更大，但训练开销也更高。

### 11.4 10 万训练数据和 50 万数据是什么关系

10 万数据必须来自 50 万候选集，不是直接从全量数据里另取。当前链路是：

```text
shibing624/medical 原始数据
  -> medical_zh_500k.jsonl
  -> cleaned_alpaca.jsonl
  -> shibing624_medical_top100k.jsonl
  -> MedicalGPT ShareGPT train.jsonl
  -> Qwen3 LoRA / QLoRA SFT
```

## 12. 本阶段验收标准

完成训练准备后，回到根项目目录检查：

```bash
cd ..
git check-ignore -v MedicalGPT/
wc -l data/sft/shibing624_medical_top100k.jsonl
wc -l MedicalGPT/data/sft_medsft_top100k/train.jsonl
```

期望：

- `MedicalGPT/` 已经被根仓库忽略。
- 根项目 10 万筛选数据是 `100000` 行。
- MedicalGPT 训练数据也是 `100000` 行。
- smoke test 能正常启动并保存 adapter。
