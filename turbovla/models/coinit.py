"""비전 타워 부분 co-initialization 공용 헬퍼.

LIBERO 학습기(turbovla/training/trainer.py)와 RoboTwin 어댑터
(starVLA/model/framework/VLM4A/TurboVLA.py)가 같은 매핑 규칙을 쓰도록 한 곳에 둔다.

GroundingDINO 체크포인트의 transformer.encoder.layers 는 MSDeformAttn 기반이라
표준 MHA 와 일대일로 대응하지 않는다.
  - linear1/linear2/norm1/norm2 : 이름이 그대로 대응 (접두사 매핑으로 처리)
  - self_attn.value_proj        : in_proj_weight/bias 의 V 구간(rows 2d:3d)
  - self_attn.output_proj       : out_proj
  - sampling_offsets/attention_weights : 대응물 없음 → 버림 (Q/K 는 랜덤으로 남음)
"""

from __future__ import annotations

VISION_TOWER_SRC_PREFIX = "transformer.encoder.layers."
VISION_TOWER_DST_PREFIX = "vision_language_interaction.vision_towers."
VISION_TOWER_PREFIX_PAIR = (VISION_TOWER_SRC_PREFIX, VISION_TOWER_DST_PREFIX)

# 사전학습 가중치를 (전부) 받지 못한 텐서. LIBERO 학습기와 같은 규칙이다.
# in_proj_* 는 Q/K/V 가 한 텐서라 통째로 여기 넣는다.
_FRESH_SUFFIXES = ("in_proj_weight", "in_proj_bias", ".gamma")


def vision_towers_of(model):
    """모델(또는 DDP/프레임워크 래퍼)에서 vision_towers ModuleList 를 찾는다. 없으면 None."""
    for candidate in (model, getattr(model, "module", None), getattr(model, "model", None)):
        if candidate is None:
            continue
        interaction = getattr(candidate, "vision_language_interaction", None)
        towers = getattr(interaction, "vision_towers", None)
        if towers is not None:
            return towers
    return None


def is_vision_tower_fresh_param(param_name: str) -> bool:
    return VISION_TOWER_DST_PREFIX in param_name and param_name.endswith(_FRESH_SUFFIXES)


def map_vision_tower_attention(source_state, target_state, vision_towers, mapped, target_prefix=""):
    """value_proj/output_proj 를 표준 MHA 자리로 옮겨 `mapped` 에 추가한다.

    target_prefix: 대상 state_dict 키 앞에 붙는 접두사 (프레임워크 래퍼가 'model.' 을
    붙이는 경우 등). 반환값은 옮긴 텐서 수.
    """
    if vision_towers is None or len(vision_towers) == 0:
        return 0
    hidden = int(vision_towers[0].hidden_dim)
    moved = 0
    for layer_idx in range(len(vision_towers)):
        src = f"{VISION_TOWER_SRC_PREFIX}{layer_idx}.self_attn."
        dst = f"{target_prefix}{VISION_TOWER_DST_PREFIX}{layer_idx}.self_attn."
        for suffix in ("weight", "bias"):
            v_key, in_key = src + f"value_proj.{suffix}", dst + f"in_proj_{suffix}"
            if v_key in source_state and in_key in target_state:
                src_v = source_state[v_key]
                new_in = mapped.get(in_key, target_state[in_key]).clone()
                if new_in[2 * hidden : 3 * hidden].shape == src_v.shape:
                    new_in[2 * hidden : 3 * hidden] = src_v
                    mapped[in_key] = new_in
                    moved += 1
            o_key, out_key = src + f"output_proj.{suffix}", dst + f"out_proj.{suffix}"
            if o_key in source_state and out_key in target_state:
                if tuple(target_state[out_key].shape) == tuple(source_state[o_key].shape):
                    mapped[out_key] = source_state[o_key]
                    moved += 1
    return moved
