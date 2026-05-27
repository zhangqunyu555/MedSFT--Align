#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3-4B-Instruct"
ADAPTER=""
TASKS="ceval-valid_clinical_medicine,ceval-valid_basic_medicine"
OUTPUT_PATH="results/ceval/qwen3_4b_instruct"
DEVICE="cuda:0"
BATCH_SIZE="auto"
DTYPE="bfloat16"
LIMIT=""
LOG_SAMPLES="true"
TRUST_REMOTE_CODE="True"
LM_EVAL_BIN="${LM_EVAL_BIN:-lm-eval}"

usage() {
  cat <<'EOF'
Run C-Eval medical evaluation with lm-evaluation-harness.

Usage:
  bash scripts/run_ceval_lm_eval.sh [options]

Options:
  --model PATH_OR_ID       Base HF model id or local model path.
  --adapter PATH           Optional PEFT/LoRA adapter path.
  --tasks TASKS            Comma-separated lm-eval task names.
  --output PATH            Output directory for lm-eval results.
  --device DEVICE          Device, e.g. cuda:0.
  --batch-size VALUE       Batch size, e.g. auto, auto:4, 1, 8.
  --dtype VALUE            Model dtype, e.g. bfloat16, float16, float32.
  --limit N                Evaluate only N samples per task for smoke test.
  --no-log-samples         Do not save per-sample logs.
  -h, --help               Show this help.

Examples:
  bash scripts/run_ceval_lm_eval.sh --limit 5
  bash scripts/run_ceval_lm_eval.sh --adapter MedicalGPT/outputs/qwen3_4b_medical_qlora_top100k
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --adapter)
      ADAPTER="$2"
      shift 2
      ;;
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --no-log-samples)
      LOG_SAMPLES="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "${LM_EVAL_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${LM_EVAL_BIN} not found. Install with: pip install -U \"lm_eval[hf]\"" >&2
  exit 1
fi

MODEL_ARGS="pretrained=${MODEL},trust_remote_code=${TRUST_REMOTE_CODE},dtype=${DTYPE}"
if [[ -n "${ADAPTER}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},peft=${ADAPTER}"
fi

mkdir -p "${OUTPUT_PATH}"

CMD=(
  "${LM_EVAL_BIN}" run
  --model hf
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${OUTPUT_PATH}"
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

if [[ "${LOG_SAMPLES}" == "true" ]]; then
  CMD+=(--log_samples)
fi

echo "Running lm-evaluation-harness C-Eval evaluation"
echo "  model: ${MODEL}"
echo "  adapter: ${ADAPTER:-<none>}"
echo "  tasks: ${TASKS}"
echo "  output: ${OUTPUT_PATH}"
echo "  device: ${DEVICE}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  limit: ${LIMIT:-<full>}"
echo
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}"
