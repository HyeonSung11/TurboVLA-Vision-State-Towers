#!/usr/bin/env bash
# 남은 3개 스위트: 재생성 -> TFDS 변환을 스위트별로 이어서 처리
# 사용: bash 14_libero_rest.sh   (conda activate 불필요)
set -uo pipefail
S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for S in libero_spatial libero_goal libero_10; do
  echo "########## [$S] 재생성 시작 $(date +%H:%M:%S) ##########"
  MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    bash "$S_DIR/11_libero_regen.sh" "$S" || { echo "[FAILED] $S 재생성"; continue; }

  echo "########## [$S] TFDS 변환 시작 $(date +%H:%M:%S) ##########"
  bash "$S_DIR/12_libero_tfds.sh" "$S" || { echo "[FAILED] $S TFDS"; continue; }
  echo "########## [$S] 완료 $(date +%H:%M:%S) ##########"
done
echo "########## 전체 완료 $(date +%H:%M:%S) ##########"
