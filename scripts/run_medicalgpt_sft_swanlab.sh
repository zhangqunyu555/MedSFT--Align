#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEDICALGPT_DIR="${ROOT_DIR}/MedicalGPT"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B-Instruct}"
TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-data/sft_medsft_top100k}"
VALIDATION_FILE_DIR="${VALIDATION_FILE_DIR:-data/sft_medsft_top100k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3_4b_medical_qlora_top100k}"
RUN_NAME="${RUN_NAME:-qwen3-4b-medical-qlora-top100k}"

MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:--1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1000}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-all}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
USE_QLORA="${USE_QLORA:-true}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-true}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-4}"

export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-MedSFT-Align}"
export SWANLAB_EXP_NAME="${SWANLAB_EXP_NAME:-${RUN_NAME}}"
export SWANLAB_TAGS="${SWANLAB_TAGS:-qwen3,sft,qlora,ceval-medical}"

usage() {
  cat <<'EOF'
Run MedicalGPT Qwen3 SFT with SwanLab logging.

This script is configured mainly through environment variables.

Common overrides:
  MODEL_NAME_OR_PATH=/path/to/Qwen3-4B-Instruct
  TRAIN_FILE_DIR=data/sft_medsft_top100k
  OUTPUT_DIR=outputs/qwen3_4b_medical_qlora_top100k
  RUN_NAME=qwen3-4b-medical-qlora-top100k
  MAX_TRAIN_SAMPLES=200
  SWANLAB_MODE=local
  SWANLAB_API_KEY=...

Usage:
  bash scripts/run_medicalgpt_sft_swanlab.sh
  MAX_TRAIN_SAMPLES=200 OUTPUT_DIR=outputs/smoke bash scripts/run_medicalgpt_sft_swanlab.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "${MEDICALGPT_DIR}" ]]; then
  echo "ERROR: MedicalGPT directory not found: ${MEDICALGPT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${MEDICALGPT_DIR}/${TRAIN_FILE_DIR}/train.jsonl" ]]; then
  cat >&2 <<EOF
ERROR: Training file not found:
  ${MEDICALGPT_DIR}/${TRAIN_FILE_DIR}/train.jsonl

Create it first with:
  cd ${MEDICALGPT_DIR}
  mkdir -p ${TRAIN_FILE_DIR}
  python tools/convert_dataset.py \\
    --in_file ../data/sft/shibing624_medical_top100k.jsonl \\
    --out_file ${TRAIN_FILE_DIR}/train.jsonl \\
    --data_type alpaca \\
    --file_type jsonl
EOF
  exit 1
fi

if ! python -c "import swanlab" >/dev/null 2>&1; then
  echo "ERROR: swanlab is not installed. Install with: pip install -U swanlab" >&2
  exit 1
fi

cd "${MEDICALGPT_DIR}"

CMD=(
  python training/supervised_finetuning.py
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --train_file_dir "${TRAIN_FILE_DIR}"
  --validation_file_dir "${VALIDATION_FILE_DIR}"
  --do_train
  --do_eval
  --use_peft True
  --max_train_samples "${MAX_TRAIN_SAMPLES}"
  --max_eval_samples "${MAX_EVAL_SAMPLES}"
  --model_max_length "${MODEL_MAX_LENGTH}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --warmup_ratio "${WARMUP_RATIO}"
  --weight_decay 0.05
  --logging_strategy steps
  --logging_steps "${LOGGING_STEPS}"
  --eval_strategy steps
  --eval_steps "${EVAL_STEPS}"
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --save_total_limit 3
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"
  --template_name qwen3
  --target_modules "${TARGET_MODULES}"
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --torch_dtype "${TORCH_DTYPE}"
  --bf16
  --gradient_checkpointing True
  --report_to swanlab
)

if [[ "${USE_QLORA}" == "true" ]]; then
  CMD+=(--qlora True)
fi

if [[ "${LOAD_IN_4BIT}" == "true" ]]; then
  CMD+=(--load_in_4bit True --optim paged_adamw_32bit)
fi

export CUDA_VISIBLE_DEVICES

echo "Running MedicalGPT SFT with SwanLab"
echo "  project: ${SWANLAB_PROJ_NAME}"
echo "  experiment: ${SWANLAB_EXP_NAME}"
echo "  tags: ${SWANLAB_TAGS}"
echo "  mode: ${SWANLAB_MODE:-cloud}"
echo "  model: ${MODEL_NAME_OR_PATH}"
echo "  train_file_dir: ${TRAIN_FILE_DIR}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  cuda: ${CUDA_VISIBLE_DEVICES}"
echo
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}"
