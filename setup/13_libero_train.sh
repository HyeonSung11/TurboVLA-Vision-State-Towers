#!/usr/bin/env bash
# [5단계] LIBERO 학습 — 4090 2장에 맞게 조정
#
# 논문 레시피: 4 GPU, global batch 256, 80k step, 10k warmup, lr 5e-5, seed 42.
#
# 저장소 기본값은 batch_size=16, grad_accum_steps=4 (trainer.py:47-48) 이며
# 4 GPU 에서 16 x 4 x 4 = 256 입니다. 여기서는 GPU 2 장이므로 accum 만 8 로
# 늘려 16 x 8 x 2 = 256 으로 동일한 global batch 를 만듭니다.
# per-device 배치가 저자 설정(16)과 같으므로 GPU 당 메모리도 동일합니다.
#
# 사용: conda activate turbovla-libero && bash 13_libero_train.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
cd "$TURBOVLA_ROOT"

D="$TURBOVLA_ROOT/data/libero"
DIRS="$D/libero_10_no_noops/1.0.0,$D/libero_goal_no_noops/1.0.0,$D/libero_object_no_noops/1.0.0,$D/libero_spatial_no_noops/1.0.0"

for p in ${DIRS//,/ }; do
  [[ -d "$p" ]] || { echo "[!] TFDS 디렉토리 없음: $p  (12_libero_tfds.sh 먼저 실행)"; exit 1; }
done
[[ -f "$TURBOVLA_INIT_CKPT" ]] || { echo "[!] GroundingDINO 체크포인트 없음: $TURBOVLA_INIT_CKPT"; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC="${NPROC:-2}"
BATCH="${BATCH:-16}"          # per-process (저자 설정과 동일)
ACCUM="${ACCUM:-8}"           # 2 x 16 x 8 = 256 (논문 global batch)

echo "[INFO] GPUs=$CUDA_VISIBLE_DEVICES procs=$NPROC batch=$BATCH accum=$ACCUM global=$((NPROC*BATCH*ACCUM))"

# conda activate 없이 실행해도 base 환경의 torchrun 이 잡히지 않도록 고정
TORCHRUN="${TORCHRUN:-$(dirname "$LIBERO_PY")/torchrun}"
[[ -x "$TORCHRUN" ]] || { echo "[!] torchrun 없음: $TORCHRUN"; exit 1; }

"$TORCHRUN" --nproc_per_node="$NPROC" experiments/libero/train.py \
  --dataset_dir "$D/libero_10_no_noops/1.0.0" \
  --dataset_dirs "$DIRS" \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --dinov3_path "$DINOV3_VITB_PATH" \
  --bert_path "$BERT_MODEL_PATH" \
  --pretrained_init_ckpt "$TURBOVLA_INIT_CKPT" \
  --checkpoint_dir "${CKPT_DIR:-$HOME/runs/turbovla/libero}" \
  --batch_size "$BATCH" \
  --grad_accum_steps "$ACCUM" \
  --max_steps "${MAX_STEPS:-80000}" \
  --warmup_steps "${WARMUP_STEPS:-10000}" \
  --lr "${LR:-5e-5}" \
  --seed "${SEED:-42}" \
  --chunk_size 12 \
  --expected_image_size 256 \
  "$@"
