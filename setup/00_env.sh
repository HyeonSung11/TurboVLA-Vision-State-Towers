#!/usr/bin/env bash
# 공통 환경변수. 다른 스크립트에서 source 해서 씁니다.
#   source ~/code/turbovla/setup/00_env.sh

export TURBOVLA_ROOT="${TURBOVLA_ROOT:-$HOME/code/turbovla/TurboVLA}"
# 에셋은 대용량이므로 여유 있는 디스크에 둡니다 (ASSET_ROOT 로 변경 가능).
export ASSET_ROOT="${ASSET_ROOT:-$HOME/assets/turbovla}"

# flash-attn 설치 때 겪은 cross-device link 에러 예방 (pip 캐시와 같은 디스크)
export TMPDIR="${TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "$TMPDIR"

# HuggingFace 캐시도 큰 디스크로
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# 백본 경로
export DINOV3_VITB_PATH="$ASSET_ROOT/dinov3-vitb16"
export DINOV3_VITL_PATH="$ASSET_ROOT/dinov3-vitl16"
export BERT_MODEL_PATH="$ASSET_ROOT/bert-base-uncased"
export TURBOVLA_INIT_CKPT="$ASSET_ROOT/groundingdino_swint_ogc.pth"
export TURBOVLA_CKPT_DIR="${TURBOVLA_CKPT_DIR:-$HOME/code/turbovla/TurboVLA/pretrained/TurboVLA}"

# 헤드리스 렌더링 (LIBERO/MuJoCo)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# LIBERO (~/.libero/config.yaml 의 datasets 값과 일치해야 함)
export LIBERO_REPO="${LIBERO_REPO:-$HOME/code/turbovla/LIBERO}"
export LIBERO_DATA="${LIBERO_DATA:-$HOME/datasets/libero/datasets}"
export LIBERO_SUITES="${LIBERO_SUITES:-libero_spatial libero_object libero_goal libero_10}"

# conda activate 없이도 동작하도록 인터프리터를 명시
export LIBERO_PY="${LIBERO_PY:-$HOME/.conda/envs/turbovla-libero/bin/python}"
export ROBOTWIN_PY="${ROBOTWIN_PY:-$HOME/.conda/envs/turbovla-robotwin/bin/python}"

mkdir -p "$ASSET_ROOT"
