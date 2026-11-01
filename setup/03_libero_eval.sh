#!/usr/bin/env bash
# LIBERO 평가. 첫 인자로 suite 이름 (기본 libero_object)
# 사용: conda activate turbovla-libero && bash 03_libero_eval.sh libero_object
#       DRY_RUN=1 bash 03_libero_eval.sh    <- 시뮬 없이 체크포인트 로딩만 검증
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
cd "$TURBOVLA_ROOT"

SUITE="${1:-libero_object}"          # libero_spatial | libero_object | libero_goal | libero_10

# HF 저장소의 실제 파일명은 README(libero_object.pth)와 다릅니다.
case "$SUITE" in
  libero_spatial) CKPT_NAME=spatial.pth ;;
  libero_object)  CKPT_NAME=object.pth  ;;
  libero_goal)    CKPT_NAME=goal.pth    ;;
  libero_10)      CKPT_NAME=long.pth    ;;
  *) echo "[!] 알 수 없는 suite: $SUITE"; exit 1 ;;
esac
CKPT="$TURBOVLA_CKPT_DIR/checkpoints/libero/$CKPT_NAME"
[[ -f "$CKPT" ]] || { echo "[!] 체크포인트 없음: $CKPT"; exit 1; }
mkdir -p outputs/evaluation

ARGS=(
  --ckpt_path "$CKPT"
  --dinov3_path "$DINOV3_VITB_PATH"
  --bert_path "$BERT_MODEL_PATH"
  --stats_path experiments/libero/configs/libero_all4_stats.json
  --stats_key libero_all4_no_noops
  --task_suite_name "$SUITE"
  --num_trials_per_task 50
  --chunk_size 12
  --num_open_loop_steps 12
  --seed 7
  --precision bf16
  --result_json_path "outputs/evaluation/${SUITE}.json"
)
[[ "${DRY_RUN:-0}" == "1" ]] && ARGS+=(--dry_run_model_load true)

python experiments/libero/evaluate.py "${ARGS[@]}"
