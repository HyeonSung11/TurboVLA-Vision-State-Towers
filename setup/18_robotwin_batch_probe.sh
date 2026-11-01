#!/usr/bin/env bash
# RoboTwin per-device 배치 한계 측정
#
# 왜: 05_robotwin_train.sh 는 per_device=12 로 학습했지만, 그 값은 실측이 아니라
#     "OOM 가능성이 있다"는 추정으로 정한 값입니다. 논문 설정은 4 GPU x 48 이고
#     우리는 2 GPU 이므로 2 x 48 x 2 = 192 가 가장 가까운 재현입니다.
#     실제로 몇까지 들어가는지 재서, 다음 학습을 논문에 더 가깝게 맞춥니다.
#
# 방법: 후보 배치마다 짧게(기본 30 step) 띄우고 OOM 여부와 최대 VRAM 을 기록.
#       메모리 정점은 backward 때 나오므로 몇 step 만 돌려도 판정됩니다.
#       결과물은 버리는 RUN_ID 로 저장하고 끝나면 지웁니다.
#
# 사용: conda activate turbovla-robotwin && bash setup/18_robotwin_batch_probe.sh
#       CANDIDATES="48 32 24" bash setup/18_robotwin_batch_probe.sh
set -uo pipefail
# cd 전에 스크립트 디렉토리를 절대경로로 고정한다 (cd 후 상대경로가 깨짐)
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SETUP_DIR/00_env.sh"
cd "$TURBOVLA_ROOT"

CANDIDATES="${CANDIDATES:-48 32 24 16}"
STEPS="${STEPS:-30}"
PROBE_ROOT="${PROBE_ROOT:-$HOME/runs/turbovla/_batch_probe}"
mkdir -p "$PROBE_ROOT"

echo "[INFO] 후보 per_device: $CANDIDATES   (각 $STEPS step)"
echo "[INFO] GPU 2 장, global batch 192 유지 (accum = 96 / per_device)"
echo

printf "%-12s %-8s %-10s %-12s %s\n" "per_device" "accum" "global" "최대VRAM" "결과"
printf "%s\n" "--------------------------------------------------------------"

for PD in $CANDIDATES; do
  # global 192 를 유지하려면 accum = 96 / PD. 나누어떨어지지 않으면 반올림.
  ACC=$(( (96 + PD - 1) / PD ))
  GLOBAL=$(( 2 * PD * ACC ))
  RID="probe_pd${PD}"
  LOG="$PROBE_ROOT/${RID}.log"
  rm -rf "${PROBE_ROOT:?}/${RID}"

  # 백그라운드로 VRAM 폴링
  MAXFILE="$PROBE_ROOT/${RID}.max"; echo 0 > "$MAXFILE"
  ( while true; do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
      cur=$(cat "$MAXFILE" 2>/dev/null || echo 0)
      [[ -n "$m" && "$m" -gt "$cur" ]] && echo "$m" > "$MAXFILE"
      sleep 2
    done ) & POLL=$!

  NUM_PROCESSES=2 \
  PER_DEVICE_BATCH_SIZE="$PD" \
  GRADIENT_ACCUMULATION_STEPS="$ACC" \
  MAX_TRAIN_STEPS="$STEPS" \
  RUN_ID="$RID" \
  RUN_ROOT_DIR="$PROBE_ROOT" \
  bash "$SETUP_DIR/05_robotwin_train.sh" > "$LOG" 2>&1
  RC=$?

  kill "$POLL" 2>/dev/null; wait "$POLL" 2>/dev/null
  MAXMEM=$(cat "$MAXFILE" 2>/dev/null || echo "?")

  if grep -qaiE "out of memory|CUDA error: out of memory|torch.cuda.OutOfMemoryError" "$LOG"; then
    VERDICT="OOM"
  elif [[ $RC -eq 0 ]]; then
    VERDICT="OK"
  else
    VERDICT="실패(rc=$RC) — 로그 확인"
  fi
  printf "%-12s %-8s %-10s %-12s %s\n" "$PD" "$ACC" "$GLOBAL" "${MAXMEM}MiB" "$VERDICT"

  rm -rf "${PROBE_ROOT:?}/${RID}"
  sleep 5    # GPU 메모리 반환 대기
done

echo
echo "로그: $PROBE_ROOT/probe_pd*.log"
echo "OK 중 가장 큰 per_device 를 05_robotwin_train.sh 의 PER_DEVICE_BATCH_SIZE 로 쓰세요."
