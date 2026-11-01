#!/usr/bin/env bash
# RoboTwin clean50 에서 비전 타워 / 상태 타워 실험을 돌린다.
# 기준선(turbovla_robotwin_clean50_55k_2gpu)과 같은 조건(55k step, 2x12x8=192, lr 5e-5,
# warmup 1000, EMA 0.999)에 플래그만 덧붙인다. 05_robotwin_train.sh 를 감싼다.
#
# 사용: conda activate turbovla-robotwin
#       MODE=coinit bash setup/20_robotwin_vision_tower.sh            # 비전 타워 + co-init
#       MODE=random bash setup/20_robotwin_vision_tower.sh            # 비전 타워, 랜덤 초기화(대조군)
#       MODE=state  bash setup/20_robotwin_vision_tower.sh            # 상태 타워만
#       MODE=combo  bash setup/20_robotwin_vision_tower.sh            # 비전(co-init) + 상태
#       MAX_TRAIN_STEPS=2 RUN_ID=_preflight_coinit MODE=coinit bash setup/20_robotwin_vision_tower.sh   # 프리플라이트
set -euo pipefail
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${MODE:-coinit}"
case "$MODE" in
  coinit) FLAGS=(--framework.interaction.vision_tower true) ;;
  random) FLAGS=(--framework.interaction.vision_tower true --framework.interaction.vision_tower_coinit false) ;;
  state)  FLAGS=(--framework.interaction.state_as_modality true) ;;
  combo)  FLAGS=(--framework.interaction.vision_tower true --framework.interaction.state_as_modality true) ;;
  *) echo "[!] MODE 는 coinit|random|state|combo"; exit 1 ;;
esac

export RUN_ID="${RUN_ID:-turbovla_robotwin_clean50_55k_2gpu_${MODE}}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-55000}"
echo "[INFO] MODE=$MODE  RUN_ID=$RUN_ID  MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS"
echo "[INFO] FLAGS=${FLAGS[*]}"
exec bash "$SETUP_DIR/05_robotwin_train.sh" "${FLAGS[@]}" "$@"
