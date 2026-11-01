#!/usr/bin/env bash
# libero_10 내려받기 (다운로더 버그 우회)
#
# download_libero_datasets.py --datasets libero_100 --use-huggingface 는
# allow_patterns="libero_100/*" 로 요청하지만 HF 저장소
# yifengzhu-hf/LIBERO-datasets 에는 libero_100/ 이 없고 libero_10/ 과
# libero_90/ 이 따로 있습니다. 그래서 libero_10 이 받혀지지 않습니다.
# TurboVLA 는 libero_10 만 쓰므로 그것만 직접 받습니다.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

HF="${ENV_BIN:-$HOME/.conda/envs/turbovla-libero/bin}/hf"
[[ -x "$HF" ]] || HF=hf

"$HF" download yifengzhu-hf/LIBERO-datasets --repo-type dataset \
  --include "libero_10/*" --local-dir "$LIBERO_DATA"

echo
echo "=== 확인 ==="
for s in $LIBERO_SUITES; do
  n=$(ls -1 "$LIBERO_DATA/$s"/*_demo.hdf5 2>/dev/null | wc -l)
  printf "  %-16s %3s 개 task hdf5\n" "$s" "$n"
done
