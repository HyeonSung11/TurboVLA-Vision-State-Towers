#!/usr/bin/env bash
# RoboTwin clean50 데이터셋 다운로드 (약 3.8 GiB)
#
# 주의: HF 저장소 StarVLA/RoboTwin-Clean 은 50개 태스크를 루트에 바로 두지만,
#       학습 레지스트리(experiments/robotwin/data_registry/data_config.py:140)는
#       ROBOTWIN_DATA_ROOT/Clean/<task_name> 을 찾습니다.
#       그래서 저장소 그대로가 아니라 Clean/ 하위로 받습니다.
#
# 사용: conda activate turbovla-robotwin && bash 04_robotwin_data.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
cd "$TURBOVLA_ROOT"

DEST="${DEST:-$HOME/datasets}"
BASE="$DEST/robotwin_clean/StarVLA_RoboTwin_Clean"
LINK="$TURBOVLA_ROOT/playground/Datasets/RoboTwin"

echo "[INFO] 대상        : $BASE/Clean"
echo "[INFO] 남은 용량   : $(df -h --output=avail "$DEST" 2>/dev/null | tail -1 || echo '(경로 생성 예정)')"

mkdir -p "$BASE/Clean" "$TURBOVLA_ROOT/playground/Datasets"
python -m pip install -U "huggingface-hub[hf_xet]==0.35.3"

hf download StarVLA/RoboTwin-Clean --repo-type dataset --local-dir "$BASE/Clean"

ln -sfn "$BASE" "$LINK"

echo
echo "=== 검증 ==="
n=$(find "$LINK/Clean" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "  태스크 디렉토리 수: $n  (50 이어야 함)"
echo "  실제 용량        : $(du -sh "$BASE" | cut -f1)"
echo
echo "다음 줄을 셸에 추가하세요:"
echo "  export ROBOTWIN_DATA_ROOT=$LINK"
