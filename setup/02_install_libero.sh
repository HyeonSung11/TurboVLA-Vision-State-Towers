#!/usr/bin/env bash
# LIBERO 시뮬레이터 설치 (turbovla-libero env 에서 실행)
# 사용: conda activate turbovla-libero && bash 02_install_libero.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

if [[ "${CONDA_DEFAULT_ENV:-}" != "turbovla-libero" ]]; then
  echo "[!] conda activate turbovla-libero 후 실행하세요."; exit 1
fi

LIBERO_DIR="${LIBERO_DIR:-$HOME/code/turbovla/LIBERO}"
[[ -d "$LIBERO_DIR" ]] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"

cd "$LIBERO_DIR"
# LIBERO requirements 가 torch 를 다시 끌어와 2.3.1 을 덮어쓸 수 있으므로 torch 계열은 제외
grep -viE '^\s*(torch|torchvision|torchaudio)\b' requirements.txt > /tmp/libero_req_no_torch.txt
pip install -r /tmp/libero_req_no_torch.txt
pip install -e .

cd - >/dev/null
echo "=== 검증 ==="
python - <<'PY'
import torch, libero
print("torch    :", torch.__version__)          # 2.3.1+cu121 이어야 함
print("libero   :", libero.__file__)
from libero.libero import benchmark
print("suites   :", list(benchmark.get_benchmark_dict().keys())[:6])
PY
