#!/usr/bin/env bash
# 백본 가중치 다운로드 (DINOv3 x2, BERT, GroundingDINO)
# TurboVLA 체크포인트는 이미 pretrained/TurboVLA 에 받혀 있으므로 제외합니다.
#
# 사용: conda activate turbovla-robotwin && bash 01_download_assets.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

HF="${ENV_BIN:-$HOME/.conda/envs/turbovla-robotwin/bin}/hf"
[[ -x "$HF" ]] || HF=hf
export HF_HUB_DISABLE_XET=1        # 429 rate limit 회피

"$HF" auth whoami >/dev/null 2>&1 || { echo "[!] hf auth login 이 필요합니다"; exit 1; }

echo "### 1/3 DINOv3 ViT-B (LIBERO용)"
"$HF" download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir "$DINOV3_VITB_PATH"

echo "### 2/3 DINOv3 ViT-L (RoboTwin용)"
"$HF" download facebook/dinov3-vitl16-pretrain-lvd1689m --local-dir "$DINOV3_VITL_PATH"

echo "### 3/3 BERT base uncased"
"$HF" download google-bert/bert-base-uncased --local-dir "$BERT_MODEL_PATH"

echo "### GroundingDINO Swin-T OGC"
[[ -f "$TURBOVLA_INIT_CKPT" ]] || curl -fL --retry 3 -o "$TURBOVLA_INIT_CKPT" \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

echo; echo "=== 확인 ==="
for f in "$DINOV3_VITB_PATH/config.json" "$DINOV3_VITL_PATH/config.json" \
         "$BERT_MODEL_PATH/config.json" "$TURBOVLA_INIT_CKPT" \
         "$TURBOVLA_CKPT_DIR/checkpoints/libero/object.pth"; do
  [[ -f "$f" ]] && echo "  OK   $f" || echo "  MISS $f"
done
