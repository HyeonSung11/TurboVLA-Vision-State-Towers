#!/usr/bin/env bash
# 우리가 학습한 RoboTwin 모델을 기준선과 같은 방식으로 평가한다 (clean50, 과제당 100회, EMA 55k).
# 사용: MODE=coinit bash setup/21_robotwin_eval_ours.sh     # coinit|random|state|combo
#       CKPT=/path/to/steps_55000_ema_pytorch_model.pt POLICY_NAME=xxx bash setup/21_robotwin_eval_ours.sh
set -euo pipefail
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SETUP_DIR/00_env.sh"
cd "$TURBOVLA_ROOT"
MODE="${MODE:-coinit}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-$HOME/runs/turbovla}"
CKPT="${CKPT:-$RUN_ROOT_DIR/turbovla_robotwin_clean50_55k_2gpu_${MODE}/checkpoints/steps_55000_ema_pytorch_model.pt}"
export POLICY_NAME="${POLICY_NAME:-turbovla_clean50_${MODE}}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-$TURBOVLA_ROOT/../RoboTwin}"
export STARVLA_PYTHON="${STARVLA_PYTHON:-$ROBOTWIN_PY}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-$HOME/.conda/envs/robotwin-sim/bin/python}"
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-100}"          # 기준선과 동일: 과제당 100회
export ROBOTWIN_JOBS_PER_GPU="${ROBOTWIN_JOBS_PER_GPU:-1}"
export ROBOTWIN_BASE_PORT="${ROBOTWIN_BASE_PORT:-7100}"
export DINOV3_MODEL_PATH="$DINOV3_VITL_PATH"; export BERT_MODEL_PATH TURBOVLA_INIT_CKPT
export PYTHONPATH="${TURBOVLA_ROOT}:${TURBOVLA_ROOT}/third_party/starvla_runtime:${PYTHONPATH:-}"
[[ -f "$CKPT" ]] || { echo "[!] 체크포인트 없음: $CKPT"; exit 1; }
[[ -d "$ROBOTWIN_PATH" ]] || { echo "[!] RoboTwin 저장소 없음: $ROBOTWIN_PATH"; exit 1; }
echo "[INFO] MODE=$MODE POLICY_NAME=$POLICY_NAME TEST_NUM=$ROBOTWIN_TEST_NUM"; echo "[INFO] CKPT=$CKPT"
bash scripts/robotwin/evaluate.sh "$CKPT" "$@"
