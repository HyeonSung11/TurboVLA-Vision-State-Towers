#!/usr/bin/env bash
# [6단계] 우리가 학습한 LIBERO 베이스라인 평가
#
# 03_libero_eval.sh 는 저자가 배포한 스위트별 체크포인트(spatial.pth 등)를 평가합니다.
# 우리 모델은 4개 스위트를 함께 학습한 단일 체크포인트이므로, 같은 가중치로
# 4개 스위트를 모두 돌립니다. 논문 기준선 숫자가 여기서 나옵니다.
#
# 사용:
#   conda activate turbovla-libero
#   bash setup/16_libero_eval_ours.sh                       # 4개 스위트 전부
#   bash setup/16_libero_eval_ours.sh libero_object         # 특정 스위트만
#   CKPT=/path/to/x.pth bash setup/16_libero_eval_ours.sh   # 다른 체크포인트
#   TRIALS=5 bash setup/16_libero_eval_ours.sh              # 빠른 점검용
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
cd "$TURBOVLA_ROOT"

CKPT="${CKPT:-$HOME/runs/turbovla/libero_baseline/turbovla_step_80000.pth}"
TRIALS="${TRIALS:-50}"
SEED="${SEED:-7}"
OUT="${OUT:-$HOME/runs/turbovla/eval/libero_baseline_80k}"
LOGDIR="$OUT/logs"

[[ -f "$CKPT" ]] || { echo "[!] 체크포인트 없음: $CKPT"; exit 1; }
mkdir -p "$OUT" "$LOGDIR"

if [[ $# -gt 0 ]]; then SUITES=("$@"); else
  SUITES=(libero_spatial libero_object libero_goal libero_10)
fi

echo "[INFO] ckpt   = $CKPT"
echo "[INFO] suites = ${SUITES[*]}"
echo "[INFO] trials = $TRIALS/task, seed=$SEED"
echo "[INFO] out    = $OUT"

run_suite () {                      # $1=suite  $2=gpu
  local suite="$1" gpu="$2"
  # 이 환경엔 OSMesa 가 없습니다(rollout.py 기본값은 osmesa). EGL 로 강제합니다 —
  # 00_env.sh 및 데이터 재생성 때 검증된 백엔드이고, GPU 렌더링이라 더 빠릅니다.
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "$LIBERO_PY" experiments/libero/evaluate.py \
    --mujoco_gl egl \
    --pyopengl_platform egl \
    --ckpt_path "$CKPT" \
    --dinov3_path "$DINOV3_VITB_PATH" \
    --bert_path "$BERT_MODEL_PATH" \
    --stats_path experiments/libero/configs/libero_all4_stats.json \
    --stats_key libero_all4_no_noops \
    --task_suite_name "$suite" \
    --num_trials_per_task "$TRIALS" \
    --chunk_size 12 \
    --num_open_loop_steps 12 \
    --seed "$SEED" \
    --precision bf16 \
    --result_json_path "$OUT/${suite}.json" \
    > "$LOGDIR/${suite}.log" 2>&1
  echo "[done] $suite (gpu $gpu) exit=$?"
}

# GPU 2장에 2개씩 짝지어 실행. osmesa 렌더링이 CPU 바운드라 4개 동시는 피합니다.
i=0
while [[ $i -lt ${#SUITES[@]} ]]; do
  run_suite "${SUITES[$i]}" 0 &
  P0=$!
  if [[ $((i+1)) -lt ${#SUITES[@]} ]]; then
    run_suite "${SUITES[$((i+1))]}" 1 &
    P1=$!
  else
    P1=""
  fi
  wait "$P0"; [[ -n "$P1" ]] && wait "$P1"
  i=$((i+2))
done

echo
echo "=== 결과 ==="
"$LIBERO_PY" - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
order = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
label = {"libero_spatial":"Spatial","libero_object":"Object","libero_goal":"Goal","libero_10":"Long"}
rates = []
for s in order:
    f = out / f"{s}.json"
    if not f.exists():
        print(f"{label[s]:9s}   (없음)"); continue
    d = json.load(open(f))
    r = d.get("success_rate", d.get("overall_success_rate"))
    if r is None:                      # 키 이름이 다르면 성공/전체로 계산
        ok, tot = d.get("num_successes"), d.get("num_episodes")
        r = ok / tot if ok is not None and tot else None
    if r is not None:
        r = r * 100 if r <= 1.0 else r
        rates.append(r)
        print(f"{label[s]:9s}   {r:5.1f}%")
    else:
        print(f"{label[s]:9s}   (파싱 실패) keys={list(d)[:8]}")
if len(rates) == 4:
    print(f"{'Average':9s}   {sum(rates)/4:5.1f}%")
PY
