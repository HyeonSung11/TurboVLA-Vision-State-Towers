#!/usr/bin/env bash
# [스모크] vision_tower 를 GPU 에서 실제로 돌려본다.
#
# 지금까지 검증은 전부 CPU 였다. CUDA 커널, bf16, SDPA flash 경로,
# DDP gradient 동기화는 실제 GPU 에서만 확인된다.
# 100 step 만 돌려 크래시/NaN/메모리를 본다.
#
# 사용: conda activate turbovla-libero && bash setup/19_smoke_vision_tower.sh
#       MODE=coinit|random|all|baseline  로 구성 선택 (기본 coinit)
set -uo pipefail
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SETUP_DIR/00_env.sh"

MODE="${MODE:-coinit}"
STEPS="${STEPS:-100}"
OUT="$HOME/runs/turbovla/_smoke_${MODE}"
rm -rf "$OUT"; mkdir -p "$OUT"

case "$MODE" in
  baseline) FLAGS=() ;;
  coinit)   FLAGS=(--vision_tower) ;;
  random)   FLAGS=(--vision_tower --no_vision_tower_coinit) ;;
  all)      FLAGS=(--vision_tower --state_as_modality --layer_tap) ;;
  *) echo "[!] MODE 는 baseline|coinit|random|all"; exit 1 ;;
esac

echo "[INFO] MODE=$MODE  STEPS=$STEPS"
echo "[INFO] FLAGS=${FLAGS[*]:-(없음)}"
echo "[INFO] OUT=$OUT"
echo

MAX_STEPS="$STEPS" WARMUP_STEPS=10 CKPT_DIR="$OUT" \
  bash "$SETUP_DIR/13_libero_train.sh" \
  --save_steps 1000000 --no_save_final --log_freq 10 "${FLAGS[@]}"
