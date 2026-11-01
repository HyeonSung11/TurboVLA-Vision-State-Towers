#!/usr/bin/env bash
# [6-B단계] 저자 배포 체크포인트를 "우리 환경"에서 평가 — 대조군
#
# 목적: 논문 수치(97.7%)와 우리 학습 결과(95.8%) 사이의 차이가
#       (a) 환경/평가 조건 차이인지  (b) 우리 학습 run 의 차이인지 분리합니다.
#       그래서 16_libero_eval_ours.sh 와 설정을 한 글자도 다르게 두지 않습니다.
#       (50 trial/task, seed 7, bf16, chunk 12, EGL, 동일 stats)
#
# 배포 체크포인트는 스위트별로 따로입니다(spatial/object/goal/long).
# 우리 체크포인트는 4개 스위트 공동 학습 단일 모델입니다 — 이 차이가 비교의 핵심입니다.
#
# 사용:
#   conda activate turbovla-libero
#   bash setup/17_libero_eval_released.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
cd "$TURBOVLA_ROOT"

TRIALS="${TRIALS:-50}"
SEED="${SEED:-7}"
OUT="${OUT:-$HOME/runs/turbovla/eval/libero_released}"
LOGDIR="$OUT/logs"
mkdir -p "$OUT" "$LOGDIR"

if [[ $# -gt 0 ]]; then SUITES=("$@"); else
  SUITES=(libero_spatial libero_object libero_goal libero_10)
fi

ckpt_for () {
  case "$1" in
    libero_spatial) echo "$TURBOVLA_CKPT_DIR/checkpoints/libero/spatial.pth" ;;
    libero_object)  echo "$TURBOVLA_CKPT_DIR/checkpoints/libero/object.pth"  ;;
    libero_goal)    echo "$TURBOVLA_CKPT_DIR/checkpoints/libero/goal.pth"    ;;
    libero_10)      echo "$TURBOVLA_CKPT_DIR/checkpoints/libero/long.pth"    ;;
  esac
}

echo "[INFO] 대조군 = 저자 배포 스위트별 체크포인트"
echo "[INFO] suites = ${SUITES[*]}  trials=$TRIALS  seed=$SEED"
echo "[INFO] out    = $OUT"

run_suite () {
  local suite="$1" gpu="$2" ckpt
  ckpt="$(ckpt_for "$suite")"
  [[ -f "$ckpt" ]] || { echo "[!] 없음: $ckpt"; return 1; }
  # rollout.py 기본값은 osmesa 인데 이 서버엔 OSMesa 가 없습니다. EGL 로 강제.
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "$LIBERO_PY" experiments/libero/evaluate.py \
    --mujoco_gl egl \
    --pyopengl_platform egl \
    --ckpt_path "$ckpt" \
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

i=0
while [[ $i -lt ${#SUITES[@]} ]]; do
  run_suite "${SUITES[$i]}" 0 & P0=$!
  if [[ $((i+1)) -lt ${#SUITES[@]} ]]; then run_suite "${SUITES[$((i+1))]}" 1 & P1=$!; else P1=""; fi
  wait "$P0"; [[ -n "$P1" ]] && wait "$P1"
  i=$((i+2))
done

echo
echo "=== 3자 비교 ==="
"$LIBERO_PY" - "$OUT" $HOME/runs/turbovla/eval/libero_baseline_80k <<'PY'
import json, sys, pathlib
rel, ours = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
order = ["libero_spatial","libero_object","libero_goal","libero_10"]
label = {"libero_spatial":"Spatial","libero_object":"Object","libero_goal":"Goal","libero_10":"Long"}
def rate(p):
    if not p.exists(): return None
    j = json.load(open(p))
    r = j.get("overall_success_rate")
    return None if r is None else r*100
print(f"{'':9s} {'배포@우리환경':>14s} {'우리학습':>10s} {'차이':>8s}")
A=[];B=[]
for s in order:
    a, b = rate(rel/f"{s}.json"), rate(ours/f"{s}.json")
    sa = f"{a:.1f}%" if a is not None else "—"
    sb = f"{b:.1f}%" if b is not None else "—"
    sd = f"{b-a:+.1f}" if (a is not None and b is not None) else "—"
    if a is not None: A.append(a)
    if b is not None: B.append(b)
    print(f"{label[s]:9s} {sa:>14s} {sb:>10s} {sd:>8s}")
if len(A)==4 and len(B)==4:
    ma, mb = sum(A)/4, sum(B)/4
    print(f"{'Average':9s} {ma:13.1f}% {mb:9.1f}% {mb-ma:+8.1f}")
    print(f"\n논문 보고값 97.7%  |  배포@우리환경 {ma:.1f}%  |  우리학습 {mb:.1f}%")
PY
