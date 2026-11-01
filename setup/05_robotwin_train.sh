#!/usr/bin/env bash
# RoboTwin clean50 학습 — 4090 2장에 맞게 조정한 버전
# 사용: conda activate turbovla-robotwin && bash 05_robotwin_train.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
cd "$TURBOVLA_ROOT"

export ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-$TURBOVLA_ROOT/playground/Datasets/RoboTwin}"
export DINOV3_MODEL_PATH="$DINOV3_VITL_PATH"     # RoboTwin 은 ViT-L
export BERT_MODEL_PATH TURBOVLA_INIT_CKPT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# 논문 레시피: global batch 192 = 4 GPU x per_device 48 x accum 1.
# GPU 2 장이므로 2 x 48 x 2 = 192 로 맞추는 것이 저자 설정과 가장 가깝습니다.
# 다만 DeepSpeed ZeRO-2 는 옵티마이저 상태를 GPU 수로 나누므로, 2 장에서는
# GPU 당 상태가 2 배가 됩니다. DINOv3 ViT-L + 3 뷰 조합이라 24GB 에서
# per_device 48 은 OOM 가능성이 있습니다. 아래 값에서 시작해 여유가 있으면
# PER_DEVICE 를 올리고 ACCUM 을 같은 배수로 줄이세요 (곱이 96 이면 global 192).
# train.sh 의 launcher_python 기본값이 "python" 이라 conda activate 없이
# 실행하면 시스템 파이썬이 잡혀 accelerate 를 못 찾습니다.
export STARVLA_PYTHON="${STARVLA_PYTHON:-$ROBOTWIN_PY}"

export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-12}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-55000}"
export RUN_ID="${RUN_ID:-turbovla_robotwin_clean50_55k_2gpu}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-$HOME/runs/turbovla}"

# train.sh 의 wandb_entity 기본값이 placeholder("your_wandb_entity") 라
# 온라인 모드면 시작 시 막힙니다. 기본을 offline 으로 둡니다.
# 온라인으로 쓰려면: WANDB_MODE=online WANDB_ENTITY=<본인엔티티> bash 05_robotwin_train.sh
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$RUN_ROOT_DIR}"
mkdir -p "$RUN_ROOT_DIR"

bash scripts/robotwin/train.sh "$@"
