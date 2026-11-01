#!/usr/bin/env bash
# [4단계] no-noops HDF5 -> TFDS/RLDS
#
# 학습 로더가 tfds.builder_from_directory() 로 읽으므로 HDF5 로는 안 됩니다
# (libero_rlds.py:188). 이 단계가 그 변환을 합니다.
#
# 사용: conda activate turbovla-libero
#       bash 12_libero_tfds.sh                  # 4개 전부
#       bash 12_libero_tfds.sh libero_object    # 하나만
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

OUT_ROOT="$TURBOVLA_ROOT/data/libero"
SUITES="${*:-$LIBERO_SUITES}"
mkdir -p "$OUT_ROOT"

for S in $SUITES; do
  SRC="$LIBERO_DATA/${S}_no_noops"
  echo "=========================================================="
  echo "[$S] $SRC -> $OUT_ROOT/${S}_no_noops/1.0.0"
  [[ -d "$SRC" ]] || { echo "  [!] 3단계 출력 없음, 건너뜀"; continue; }
  "$LIBERO_PY" "$(dirname "${BASH_SOURCE[0]}")/libero_hdf5_to_rlds.py" \
    --suite "$S" --hdf5_dir "$SRC" --out_root "$OUT_ROOT" --overwrite
done

echo; echo "=== 최종 레이아웃 ==="
find "$OUT_ROOT" -maxdepth 2 -name "1.0.0" | sort
