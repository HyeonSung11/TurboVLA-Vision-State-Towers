#!/usr/bin/env bash
# RoboTwin 학습 재개 (서버 재부팅 등으로 중단된 경우)
#
# scripts/robotwin/train.sh 는 출력 디렉토리가 이미 있으면 종료하므로
# 재개할 때는 쓸 수 없습니다. 이 스크립트는 train.sh 와 동일한 실행 명령을
# 재현하되 --trainer.is_resume true 를 추가합니다.
#
# 재개 시 복원되는 것 / 안 되는 것 (train_starvla.py:244-258 기준):
#   복원 O : 모델 가중치(steps_<N>_pytorch_model.pt), LR 스케줄러 위치
#   복원 X : 옵티마이저 상태(Adam moment), EMA 상태, 데이터로더 순서
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
cd "$TURBOVLA_ROOT"

export PYTHONPATH="${TURBOVLA_ROOT}:${TURBOVLA_ROOT}/third_party/starvla_runtime:${PYTHONPATH:-}"
export ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-$TURBOVLA_ROOT/playground/Datasets/RoboTwin}"
export DINOV3_MODEL_PATH="$DINOV3_VITL_PATH"
export BERT_MODEL_PATH TURBOVLA_INIT_CKPT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_IB_DISABLE=1 NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1

RUN_ROOT_DIR="${RUN_ROOT_DIR:-$HOME/runs/turbovla}"
RUN_ID="${RUN_ID:-turbovla_robotwin_clean50_55k_2gpu}"
OUT="$RUN_ROOT_DIR/$RUN_ID"
[[ -d "$OUT/checkpoints" ]] || { echo "[!] 체크포인트 디렉토리 없음: $OUT/checkpoints"; exit 1; }
echo "[INFO] 재개 대상: $(ls -1 "$OUT/checkpoints" | grep -oE 'steps_[0-9]+_pytorch_model\.pt' | sort -t_ -k2 -n | tail -1)"

NUM_PROCESSES="${NUM_PROCESSES:-2}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-12}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-55000}"
LR="${LEARNING_RATE:-5.0e-05}"
echo "[INFO] processes=$NUM_PROCESSES per_device=$PER_DEVICE_BATCH_SIZE accum=$GRADIENT_ACCUMULATION_STEPS global=$((NUM_PROCESSES*PER_DEVICE_BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS))"

"$ROBOTWIN_PY" -m accelerate.commands.launch \
  --config_file experiments/robotwin/configs/deepspeed_zero2.yaml \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port "${MAIN_PROCESS_PORT:-29630}" \
  third_party/starvla_runtime/starVLA/training/train_robotwin_clean_act_pi05_recipe.py \
  --config_yaml experiments/robotwin/configs/clean50.yaml \
  --datasets.vla_data.per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --trainer.learning_rate.base "$LR" \
  --trainer.learning_rate.text_encoder "$LR" \
  --trainer.learning_rate.vision_encoder "$LR" \
  --trainer.learning_rate.vision_language_interaction "$LR" \
  --trainer.learning_rate.vision_projection "$LR" \
  --trainer.learning_rate.action_head "$LR" \
  --trainer.gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --trainer.ema_decay "${EMA_DECAY:-0.999}" \
  --trainer.ema_device "${EMA_DEVICE:-cuda}" \
  --trainer.max_train_steps "$MAX_TRAIN_STEPS" \
  --trainer.num_warmup_steps "${WARMUP_STEPS:-1000}" \
  --trainer.save_interval "${SAVE_INTERVAL:-5000}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY:-50}" \
  --trainer.is_resume true \
  --run_root_dir "$RUN_ROOT_DIR" \
  --run_id "$RUN_ID" \
  --wandb_project "${WANDB_PROJECT:-turbovla_robotwin}" \
  --wandb_entity "${WANDB_ENTITY:-turbovla}" \
  "$@"
