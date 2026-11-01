#!/usr/bin/env bash
# [3단계] no-op 제거 + 256x256 재렌더링
#
# 원본 LIBERO 데모는 128x128 (env_wrapper.py: camera_heights=128) 인데
# TurboVLA 로더는 256x256 을 강제하고 리사이즈를 거부합니다
# (libero_rlds.py:123-129). 이 스크립트가 시뮬레이터로 데모를 재생하며
# 256 으로 다시 렌더링하고, 동시에 no-op 프레임을 제거합니다.
#
# 사용: conda activate turbovla-libero
#       bash 11_libero_regen.sh                  # 4개 스위트 전부
#       bash 11_libero_regen.sh libero_object    # 하나만
#
# GPU: MuJoCo EGL 렌더링에 GPU 를 씁니다. 다른 작업과 경합하지 않도록
#      MUJOCO_EGL_DEVICE_ID / CUDA_VISIBLE_DEVICES 를 지정하세요.
#      예) MUJOCO_EGL_DEVICE_ID=1 bash 11_libero_regen.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
cd "$TURBOVLA_ROOT"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

SUITES="${*:-$LIBERO_SUITES}"

for S in $SUITES; do
  SRC="$LIBERO_DATA/$S"
  DST="$LIBERO_DATA/${S}_no_noops"
  echo "=========================================================="
  echo "[$S] $SRC -> $DST   (EGL GPU=$MUJOCO_EGL_DEVICE_ID)"
  [[ -d "$SRC" ]] || { echo "  [!] 원본 없음, 건너뜀"; continue; }
  mkdir -p "$DST"
  "$LIBERO_PY" scripts/libero/regenerate_libero_no_noops.py \
    --libero_task_suite "$S" \
    --libero_raw_data_dir "$SRC" \
    --libero_target_dir "$DST"
  echo "[$S] 완료: $(ls -1 "$DST"/*_demo.hdf5 2>/dev/null | wc -l) 개 파일, $(du -sh "$DST" | cut -f1)"
done
