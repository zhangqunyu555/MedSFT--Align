# lm-evaluation-harness C-Eval 评估与 SwanLab 记录

本阶段解决两件事：

1. 在虚拟机上从零安装 `lm-evaluation-harness`，并用它评估 `Qwen3-4B-Instruct` 在 C-Eval 医学科目上的准确率。
2. 给 `MedicalGPT` 的 SFT 训练打开 SwanLab 记录，方便在训练时看 loss、eval loss、学习率和运行配置。

我已经检查过当前 `MedicalGPT/` 子项目：没有发现 `swanlab` 相关代码，训练脚本主要通过 `--report_to tensorboard` 记录日志。不过 `MedicalGPT` 使用的是 HuggingFace `Trainer`，所以只要安装 `swanlab`，并把训练参数改成 `--report_to swanlab`，就可以直接接入。

## 1. 准备虚拟机环境

建议使用 NVIDIA CUDA 虚拟机。完整 Qwen3-4B 评估和 QLoRA 训练都不建议放在 Mac / MPS 上跑。

先确认 GPU：

```bash
nvidia-smi
```

建议新建虚拟环境：

```bash
conda create -n medsft-align python=3.10 -y
conda activate medsft-align
```

安装 PyTorch 时按你虚拟机 CUDA 版本选择。例如 CUDA 12.1：

```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装 MedicalGPT 训练依赖：

```bash
cd MedicalGPT
pip install -r requirements.txt
cd ..
```

安装评估和 SwanLab：

```bash
pip install -U "lm_eval[hf]"
pip install -U swanlab
pip install -U transformers peft accelerate sentencepiece
```

验证安装：

```bash
lm-eval -h
python -c "import swanlab; print(swanlab.__version__)"
```

如果你的 `lm-eval` 版本较新，官方入口是：

```bash
lm-eval run ...
```

你看到的旧教程：

```bash
python main.py --model hf-causal ...
```

属于旧版 harness 的命令风格。现在优先使用新版 CLI。

## 2. 确认 C-Eval 任务名

不同版本 harness 里的任务名可能不完全一致，所以不要死记任务名，先查：

```bash
lm-eval ls tasks | grep -Ei "ceval|clinical|basic"
```

本项目主评估只关心医学相关两科：

| 中文科目 | C-Eval subject |
| --- | --- |
| 临床医学 | `clinical_medicine` |
| 基础医学 | `basic_medicine` |

如果你的任务列表里是新版命名，通常会接近：

```text
ceval-valid_clinical_medicine
ceval-valid_basic_medicine
```

如果你安装的是旧版或 C-Eval 原始适配，可能是：

```text
Ceval-valid-clinical_medicine
Ceval-valid-basic_medicine
```

脚本默认使用新版风格。若你本机任务名不同，用 `--tasks` 覆盖即可。

## 3. 评估原始 Qwen3-4B-Instruct

先跑 5 条 smoke test：

```bash
bash scripts/run_ceval_lm_eval.sh --limit 5
```

这个命令默认等价于：

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=Qwen/Qwen3-4B-Instruct,trust_remote_code=True,dtype=bfloat16 \
  --tasks ceval-valid_clinical_medicine,ceval-valid_basic_medicine \
  --device cuda:0 \
  --batch_size auto \
  --output_path results/ceval/qwen3_4b_instruct \
  --limit 5 \
  --log_samples
```

如果 smoke test 能正常下载模型、加载任务并输出结果，再跑完整评估：

```bash
bash scripts/run_ceval_lm_eval.sh
```

输出会保存在：

```text
results/ceval/qwen3_4b_instruct/
```

重点看结果中的：

- `acc`
- `acc_norm`

C-Eval 是选择题评估，准确率就是最关键指标。

## 4. 评估 LoRA / QLoRA adapter

如果你已经用 MedicalGPT 训练出了 adapter，例如：

```text
MedicalGPT/outputs/qwen3_4b_medical_qlora_top100k/
```

先 smoke test：

```bash
bash scripts/run_ceval_lm_eval.sh \
  --adapter MedicalGPT/outputs/qwen3_4b_medical_qlora_top100k \
  --output results/ceval/qwen3_4b_medical_qlora_top100k_smoke \
  --limit 5
```

正式评估：

```bash
bash scripts/run_ceval_lm_eval.sh \
  --adapter MedicalGPT/outputs/qwen3_4b_medical_qlora_top100k \
  --output results/ceval/qwen3_4b_medical_qlora_top100k
```

脚本会把 adapter 传给 harness：

```text
peft=MedicalGPT/outputs/qwen3_4b_medical_qlora_top100k
```

因此不一定要先合并 LoRA。评估时会加载：

- base model：`Qwen/Qwen3-4B-Instruct`
- adapter：`MedicalGPT/outputs/qwen3_4b_medical_qlora_top100k`

如果你已经合并成完整模型，也可以直接评估合并目录：

```bash
bash scripts/run_ceval_lm_eval.sh \
  --model /path/to/merged_qwen3_medical_model \
  --output results/ceval/qwen3_4b_medical_merged
```

## 5. 任务名不匹配怎么办

如果脚本报错说找不到任务，先运行：

```bash
lm-eval ls tasks | grep -Ei "ceval|clinical|basic"
```

假设你看到的是：

```text
Ceval-valid-clinical_medicine
Ceval-valid-basic_medicine
```

就这样运行：

```bash
bash scripts/run_ceval_lm_eval.sh \
  --tasks Ceval-valid-clinical_medicine,Ceval-valid-basic_medicine \
  --limit 5
```

原则是：以你虚拟机里 `lm-eval ls tasks` 的实际输出为准。

## 6. SwanLab 云端记录

MedicalGPT 本身没有写 SwanLab 专用代码，但它使用 HuggingFace `Trainer`。SwanLab 对新版 Transformers 已经支持 `report_to="swanlab"`，所以我们只需要：

1. 安装 SwanLab。
2. 登录 SwanLab。
3. 训练时加 `--report_to swanlab`。

登录：

```bash
swanlab login
```

或者在虚拟机上设置 API Key：

```bash
export SWANLAB_API_KEY="你的 API Key"
```

启动训练：

```bash
bash scripts/run_medicalgpt_sft_swanlab.sh
```

这个脚本会设置：

```bash
export SWANLAB_PROJ_NAME="MedSFT-Align"
export SWANLAB_EXP_NAME="qwen3-4b-medical-qlora-top100k"
export SWANLAB_TAGS="qwen3,sft,qlora,ceval-medical"
```

并在 MedicalGPT 训练命令中加入：

```bash
--report_to swanlab
--run_name qwen3-4b-medical-qlora-top100k
```

如果你想先做小样本训练验证：

```bash
MAX_TRAIN_SAMPLES=200 \
MAX_EVAL_SAMPLES=20 \
OUTPUT_DIR=outputs/qwen3_4b_medical_swanlab_smoke \
RUN_NAME=qwen3-4b-medical-swanlab-smoke \
bash scripts/run_medicalgpt_sft_swanlab.sh
```

## 7. SwanLab 离线看板

如果虚拟机不能联网，使用 local 模式：

```bash
export SWANLAB_MODE=local
```

然后启动训练：

```bash
bash scripts/run_medicalgpt_sft_swanlab.sh
```

训练会把日志写到本地 `swanlog/`。打开看板：

```bash
swanlab watch -h 0.0.0.0 -p 5092
```

如果虚拟机是远程服务器，你可以在本地做端口转发：

```bash
ssh -L 5092:127.0.0.1:5092 user@server_ip
```

然后浏览器打开：

```text
http://127.0.0.1:5092
```

如果提示没有 dashboard 依赖：

```bash
pip install -U "swanlab[dashboard]"
```

## 8. 训练前的数据准备

SwanLab 训练脚本默认读取：

```text
MedicalGPT/data/sft_medsft_top100k/train.jsonl
```

如果这个文件不存在，先从根项目 10 万 Alpaca 数据转换：

```bash
cd MedicalGPT
mkdir -p data/sft_medsft_top100k

python tools/convert_dataset.py \
  --in_file ../data/sft/shibing624_medical_top100k.jsonl \
  --out_file data/sft_medsft_top100k/train.jsonl \
  --data_type alpaca \
  --file_type jsonl

wc -l data/sft_medsft_top100k/train.jsonl
cd ..
```

期望行数：

```text
100000
```

## 9. 推荐实验顺序

建议按这个顺序跑：

1. `lm-eval` smoke test 原始 Qwen3：

```bash
bash scripts/run_ceval_lm_eval.sh --limit 5
```

2. MedicalGPT + SwanLab smoke test：

```bash
MAX_TRAIN_SAMPLES=200 \
MAX_EVAL_SAMPLES=20 \
OUTPUT_DIR=outputs/qwen3_4b_medical_swanlab_smoke \
RUN_NAME=qwen3-4b-medical-swanlab-smoke \
bash scripts/run_medicalgpt_sft_swanlab.sh
```

3. 正式 QLoRA SFT：

```bash
bash scripts/run_medicalgpt_sft_swanlab.sh
```

4. 评估 SFT adapter：

```bash
bash scripts/run_ceval_lm_eval.sh \
  --adapter MedicalGPT/outputs/qwen3_4b_medical_qlora_top100k \
  --output results/ceval/qwen3_4b_medical_qlora_top100k
```

5. 对比结果：

```text
results/ceval/qwen3_4b_instruct
results/ceval/qwen3_4b_medical_qlora_top100k
```

目标是观察 C-Eval 医学准确率是否接近项目预期：

```text
0.8324 -> 0.8652
```

## 10. 本阶段验收

脚本语法检查：

```bash
bash -n scripts/run_ceval_lm_eval.sh
bash -n scripts/run_medicalgpt_sft_swanlab.sh
```

Git 忽略检查：

```bash
git check-ignore -v results/
git check-ignore -v swanlog/
```

安装检查：

```bash
lm-eval -h
python -c "import swanlab; print(swanlab.__version__)"
```

训练检查：

- SwanLab 页面或本地看板能看到训练 loss。
- `MedicalGPT/outputs/...` 里生成 adapter。
- C-Eval 输出目录里有结果 JSON。

