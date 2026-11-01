#!/usr/bin/env bash
# RoboTwin 데이터 이어받기 — 429 rate limit 자동 재시도
# 사용: conda activate turbovla-robotwin && bash 04b_resume_download.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

BASE=$HOME/datasets/robotwin_clean/StarVLA_RoboTwin_Clean
TOTAL_TASKS=50

# xet 를 끄면 파일별 xet-read-token API 호출이 사라져 429 를 크게 줄입니다.
export HF_HUB_DISABLE_XET=1

for attempt in $(seq 1 20); do
  done_tasks=$(find "$BASE" -mindepth 1 -maxdepth 1 -type d -not -name ".cache" 2>/dev/null | wc -l)
  echo "=== 시도 $attempt | 완료 태스크 $done_tasks/$TOTAL_TASKS | $(du -sh "$BASE" 2>/dev/null | cut -f1) ==="

  if hf download StarVLA/RoboTwin-Clean --repo-type dataset \
       --local-dir "$BASE" --max-workers 4; then
    echo "[OK] 다운로드 완료"
    break
  fi

  echo "[!] 실패 — 5분 대기 후 재시도 (rate limit 창 리셋)"
  sleep 300
done

echo
echo "=== 최종 상태 ==="
echo "  태스크 : $(find "$BASE" -mindepth 1 -maxdepth 1 -type d -not -name .cache | wc -l)/50"
echo "  파일   : $(find "$BASE" -type f -not -path '*/.cache/*' | wc -l)"
echo "  용량   : $(du -sh "$BASE" | cut -f1)   (기대값 약 3.7G)"
