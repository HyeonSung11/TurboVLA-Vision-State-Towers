# Setup

Tested on Ubuntu 22.04, CUDA 12.x, Python 3.10, PyTorch 2.6 with two NVIDIA RTX 4090 (24 GB). Every script in `setup/` is numbered in the order it was used; all of them `source setup/00_env.sh`, so paths are changed there (or exported before running) rather than inside the scripts. Comments inside the scripts are in Korean.

## 1. Environments

Two conda environments are used, exactly as in the original TurboVLA repository (see [`docs/README_TurboVLA.md`](docs/README_TurboVLA.md) for the full dependency list):

| Env | Used for | Python |
|---|---|---|
| `turbovla-libero` | LIBERO data generation, training, evaluation | `$LIBERO_PY` |
| `turbovla-robotwin` | RoboTwin training (starVLA runtime, DeepSpeed ZeRO-2) and policy server | `$ROBOTWIN_PY` |

A third environment with the RoboTwin 2.0 simulator (`robotwin-sim`, `$ROBOTWIN_PYTHON`) is needed only for RoboTwin evaluation.

```bash
conda create -n turbovla-libero python=3.10 -y
conda activate turbovla-libero
pip install -e .                      # this repository
bash setup/02_install_libero.sh       # LIBERO simulator (git clone into $LIBERO_REPO)

conda create -n turbovla-robotwin python=3.10 -y
conda activate turbovla-robotwin
pip install -e . && pip install -r scripts/robotwin/requirements.txt
```

flash-attn is optional; the interaction module falls back to PyTorch SDPA.

## 2. Paths (`setup/00_env.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `TURBOVLA_ROOT` | `$HOME/code/turbovla/TurboVLA` | this repository |
| `ASSET_ROOT` | `$HOME/assets/turbovla` | backbones and the Grounding DINO checkpoint |
| `LIBERO_REPO` / `LIBERO_DATA` | `$HOME/code/turbovla/LIBERO`, `$HOME/datasets/libero/datasets` | LIBERO simulator and raw demos |
| `RUN_ROOT_DIR` | `$HOME/runs/turbovla` | training outputs and evaluation results |
| `LIBERO_PY`, `ROBOTWIN_PY`, `ROBOTWIN_PYTHON` | conda env interpreters | used so scripts run without `conda activate` |

## 3. Pretrained weights

```bash
hf auth login
bash setup/01_download_assets.sh
```

| File | Purpose | Source |
|---|---|---|
| `dinov3-vitb16/` | vision backbone (LIBERO) | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| `dinov3-vitl16/` | vision backbone (RoboTwin) | `facebook/dinov3-vitl16-pretrain-lvd1689m` |
| `bert-base-uncased/` | text encoder (frozen) | `google-bert/bert-base-uncased` |
| `groundingdino_swint_ogc.pth` | initialization of the interaction module **and of the vision tower** | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/releases/tag/v0.1.0-alpha) |

All of them go under `$ASSET_ROOT` (git-ignored).

## 4. LIBERO

Data: raw demos → no-op removal + 256×256 re-rendering → TFDS/RLDS, one suite at a time.

```bash
conda activate turbovla-libero
bash setup/10_libero_fix_libero10.sh      # downloads libero_10 (the stock downloader misses it)
bash setup/11_libero_regen.sh libero_object
bash setup/12_libero_tfds.sh  libero_object
bash setup/14_libero_rest.sh              # the remaining three suites
```

Training (one model on all four suites; global batch 256 = 2 GPUs × 16 × accum 8; 80k steps, warmup 10k, lr 5e-5, seed 42):

```bash
bash setup/13_libero_train.sh                                              # Baseline
bash setup/13_libero_train.sh --vision_tower                               # VT (co-init)
bash setup/13_libero_train.sh --vision_tower --no_vision_tower_coinit      # VT (random)
bash setup/13_libero_train.sh --state_as_modality                          # ST
bash setup/13_libero_train.sh --vision_tower --state_as_modality           # VT (co-init) + ST
# 8k-step runs of Table 1 (separate schedule, warmup 1k):
MAX_STEPS=8000 WARMUP_STEPS=1000 bash setup/13_libero_train.sh --vision_tower
```

Set `CKPT_DIR` to a different directory per configuration. Evaluation (50 trials × 10 tasks per suite, bf16, EGL rendering, evaluation seed 7):

```bash
CKPT=$RUN_ROOT_DIR/libero_vs_80k/turbovla_step_80000.pth OUT=$RUN_ROOT_DIR/eval/libero_vs_80k bash setup/16_libero_eval_ours.sh
bash setup/17_libero_eval_released.sh     # the authors' released checkpoints under the same protocol (control)
```

## 5. RoboTwin 2.0

Data: the 50-task clean dataset (`StarVLA/RoboTwin-Clean`, ≈3.8 GiB) placed under `playground/Datasets/RoboTwin/Clean/<task>`:

```bash
conda activate turbovla-robotwin
bash setup/04_robotwin_data.sh            # 04b_resume_download.sh resumes after HF rate limits
```

Training (DINOv3 ViT-L, 3 views, global batch 192 = 2 GPUs × 12 × accum 8; 55k steps, warmup 1k, lr 5e-5, EMA 0.999; wandb offline by default):

```bash
bash setup/05_robotwin_train.sh                     # Baseline (seed 42)
MODE=coinit bash setup/20_robotwin_vision_tower.sh  # VT (co-init)
MODE=random bash setup/20_robotwin_vision_tower.sh  # VT (random)
MODE=state  bash setup/20_robotwin_vision_tower.sh  # ST
MODE=combo  bash setup/20_robotwin_vision_tower.sh  # VT (co-init) + ST
# second seed of the baseline and of the combination:
RUN_ID=turbovla_robotwin_clean50_55k_2gpu_base_seed7  bash setup/05_robotwin_train.sh --seed 7
RUN_ID=turbovla_robotwin_clean50_55k_2gpu_combo_seed7 bash setup/05_robotwin_train.sh --seed 7 \
    --framework.interaction.vision_tower true --framework.interaction.state_as_modality true
```

`setup/15_robotwin_resume.sh` restarts an interrupted run from its last checkpoint (weights and LR schedule are restored; optimizer and EMA state are not). `setup/18_robotwin_batch_probe.sh` measures the largest per-device batch that fits.

Evaluation (clean setting, 100 trials per task, EMA checkpoint at 55k; needs the RoboTwin 2.0 repository at `$ROBOTWIN_PATH` and the simulator environment):

```bash
MODE=combo bash setup/21_robotwin_eval_ours.sh
CKPT=$RUN_ROOT_DIR/turbovla_robotwin_clean50_55k_2gpu_combo_seed7/checkpoints/steps_55000_ema_pytorch_model.pt \
    POLICY_NAME=turbovla_clean50_combo_seed7 bash setup/21_robotwin_eval_ours.sh
```

Per-task results are written under `$ROBOTWIN_PATH/eval_result/<POLICY_NAME>/`.

## 6. Smoke tests

`setup/19_smoke_vision_tower.sh` runs 100 LIBERO steps with the vision tower on GPU to check for crashes, NaNs and memory before a long run. With all flags off, the model's outputs are identical to the original TurboVLA.
