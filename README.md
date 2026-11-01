# Design and Analysis of Visual Self-Attention and State Towers for a Lightweight Vision-Language-Action Model
**경량 비전-언어-행동 모델을 위한 시각 셀프 어텐션 타워와 상태 타워의 설계 및 분석**

Hyeon-Sung Choi, Jong-Eun Ha*  
Seoul National University of Science and Technology

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-2563eb)](LICENSE)

> 🚀 This project is built on **[TurboVLA](https://github.com/H-EmbodVis/TurboVLA)** (Xie et al., 2026), a lightweight VLA that runs at 32 Hz on an RTX 4090 with < 1 GB VRAM.  
> The original TurboVLA README is kept at [`docs/README_TurboVLA.md`](docs/README_TurboVLA.md).

---

## Abstract

TurboVLA is a lightweight vision-language-action (VLA) model that removes the large language model and fuses visual and instruction tokens through a shallow bidirectional interaction module initialized from the feature enhancer of the open-vocabulary detector Grounding DINO. Inside this module visual tokens never attend to each other, and the robot state enters only at the action decoder. We propose two extensions: a **vision tower**, a visual self-attention block added to every interaction layer and partially co-initialized from the visual self-attention block of the same feature enhancer, and a **state tower** that injects the robot state into the fusion stage through asymmetric attention. The vision tower is wrapped by a zero-initialized gate so that enabling it leaves the model bit-identical to the baseline at the start of training, and a randomly initialized vision tower serves as a control that separates the effect of the architecture from that of weight transfer. On LIBERO, all five configurations converge to 95.4–96.2% after 80k steps, while co-initialization leads the random control by 4.2%p at 8k steps, indicating faster convergence without a change of the end point. On RoboTwin 2.0 (50 bimanual tasks, 100 trials per task), the vision tower scores 2–3%p below the seed-0 baseline, the state tower is neutral, and their combination is the highest configuration in both of its seeds (59.3% and 59.0%). A second baseline seed moves the baseline average by 4.5%p (57.6% → 53.2%), so all differences are read against this range: both combination seeds exceed both baseline seeds (+3.8%p between the two-seed means, a ranking that four exchangeable runs would produce with probability 1/6), while task-level differences stay within seed variation. A qualitative comparison on identical initial layouts shows that the same architecture can succeed or fail depending on initialization.

## Keywords

vision-language-action model · lightweight VLA · co-initialization · asymmetric attention · LIBERO · RoboTwin

---

## Method

<p align="center">
  <img src="figs/fig1_architecture.png" width="95%">
  <br>
  <em>Fig. 1. Proposed TurboVLA architecture with the vision and state towers.</em>
</p>

<p align="center">
  <img src="figs/fig2_interaction_layer.png" width="80%">
  <br>
  <em>Fig. 2. One layer of the vision-language interaction module. The original layer has text self-attention, bidirectional cross-attention and a text FFN only; the vision tower (visual self-attention + FFN, gated) and the state tower (state → vision/text asymmetric attention) are added per layer.</em>
</p>

<p align="center">
  <img src="figs/fig3_weight_transfer.png" width="80%">
  <br>
  <em>Fig. 3. Weight transfer from the Grounding DINO feature enhancer to the TurboVLA interaction module. The feature enhancer's visual self-attention is deformable, so only the value/output projections, FFN and norms are copied; query/key projections stay random.</em>
</p>

<p align="center">
  <img src="figs/fig4_token_reference.png" width="55%">
  <br>
  <em>Fig. 4. Token reference pattern in one interaction layer. (a) Baseline, (b) proposed.</em>
</p>

**Vision tower.** The original interaction module has text self-attention, bidirectional vision–text cross-attention and a text FFN, but no visual self-attention: visual tokens never attend to each other in any of the six layers. The vision tower adds a standard multi-head visual self-attention block with an FFN of width 2048 to every layer. Its output enters through a zero-initialized gate (γ = 0), so at the start of training the model is bit-identical to the baseline and the tower is learned only as far as the data demand it.

**Partial co-initialization.** The Grounding DINO feature enhancer, from which TurboVLA initializes its interaction module, also contains a visual self-attention block per layer that TurboVLA never loads. Its FFN, normalization layers and value/output projections are copied into the vision tower. The block is deformable attention, so the sampling offsets and attention weights have no counterpart in standard attention and are dropped; the query and key projections therefore remain random. A vision tower of the same shape with fully random initialization serves as the control that separates the effect of the architecture from that of the transferred weights.

**State tower.** The robot state is projected to tokens and injected at the fusion stage instead of only at the action decoder. In every layer the state tokens attend to the visual and text tokens, while the pretrained visual and text streams never attend to the state (asymmetric attention), so the pretrained representations are not disturbed. The updated state tokens are then passed to the action decoder together with the fused tokens.

**Learning rate.** Parameters that received no pretrained weights (the query/key projections of the vision tower and its gate) use five times the base learning rate, following Cosmos 3; everything else, including the state tower, uses the base learning rate. The multiplier and the FFN width were not tuned.

The five configurations compared in the paper are **Baseline**, **VT (co-init)**, **VT (random)**, **ST**, and **VT (co-init) + ST**. With both towers off the model is the original TurboVLA.

---

## Results

All runs use two RTX 4090 (24 GB) GPUs. LIBERO: one model on the four suites (spatial/object/goal/long), global batch 256, 80k steps, 50 trials × 10 tasks per suite (2,000 episodes per configuration). RoboTwin 2.0: 50 bimanual tasks (clean setting), DINOv3 ViT-L backbone, 55k steps, 100 trials per task (5,000 episodes per configuration). Following the original TurboVLA paper, only success rates are reported (no significance tests); on RoboTwin the baseline and the combination were trained twice (seeds 42 and 7; the paper calls them seed 0 and seed 7) to show the size of seed variation.

**Table 1. LIBERO success rate at 8k steps (%)** – co-initialization vs. the random control

| Suite | Baseline | VT (co-init) | VT (random) | co-init − random |
|---|---:|---:|---:|---:|
| spatial | 80.0 | 86.0 | 79.0 | +7.0 |
| object | 88.8 | 95.0 | 97.6 | −2.6 |
| goal | 68.0 | 79.6 | 73.2 | +6.4 |
| long | 42.4 | 58.0 | 52.0 | +6.0 |
| **Average** | 69.8 | **79.7** | 75.5 | **+4.2** |

**Table 2. LIBERO success rate at 80k steps (%)** (Paper = numbers reported in the TurboVLA paper, 4 GPUs)

| Suite | Paper | Baseline | VT (co-init) | VT (random) | ST | VT (co-init) + ST |
|---|---:|---:|---:|---:|---:|---:|
| spatial | 99.2 | 97.0 | 98.6 | 98.0 | 96.8 | 96.8 |
| object | 99.8 | 100.0 | 99.6 | 99.6 | 99.8 | 99.2 |
| goal | 97.4 | 93.6 | 96.6 | 93.2 | 94.2 | 97.6 |
| long | 94.2 | 92.6 | 90.0 | 93.6 | 90.8 | 89.2 |
| **Average** | 97.7 | 95.8 | 96.2 | 96.1 | 95.4 | 95.7 |

<p align="center">
  <img src="figs/fig7_libero_success.png" width="60%">
  <br>
  <em>Fig. 7. LIBERO success rate per suite. (a) 8k steps, (b) 80k steps.</em>
</p>

**Table 3. RoboTwin 2.0 success rate (%)** – W/L/T compares the seed-42 run of each configuration with the two baseline seeds task by task

| Configuration | Seed 42 | Seed 7 | Avg. | W/L/T vs. baseline seed 42 | W/L/T vs. baseline seed 7 |
|---|---:|---:|---:|---:|---:|
| Baseline | 57.6 | 53.2 | 55.4 | — | — |
| VT (co-init) | 54.4 | — | — | 14/33/3 | 25/21/4 |
| VT (random) | 55.7 | — | — | 17/31/2 | 23/26/1 |
| ST | 57.8 | — | — | 28/21/1 | 30/16/4 |
| **VT (co-init) + ST** | **59.3** | **59.0** | **59.2** | 27/18/5 | 34/14/2 |

<p align="center">
  <img src="figs/fig8_robotwin_per_task_diff.png" width="95%">
  <br>
  <em>Fig. 8. Per-task success-rate difference on RoboTwin 2.0 (relative to the seed-42 baseline). The grey bars (baseline seed 7 − seed 42) show how much a seed change alone moves each task.</em>
</p>

<p align="center">
  <img src="figs/fig9_qualitative.png" width="95%">
  <br>
  <em>Fig. 9. Qualitative comparison on RoboTwin 2.0 with identical initial layouts. (a) stack_blocks_three, (b) hanging_mug.</em>
</p>

**Summary.** Partial co-initialization speeds up the early phase (+4.2%p at 8k) but does not change the convergence point. Either tower alone does not move the final success rate. The combination is the best configuration on RoboTwin 2.0 in both seeds and exceeds both baseline seeds (+3.8%p between two-seed means); its task-level gains, however, move with the seed (task-level SD between the two combination seeds is 16.9%p), so no claim is made about particular task types.
