from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class TextEncoderConfig:
    model_name_or_path: str = "bert-base-uncased"
    max_length: int = 256
    padding_length: int | None = None
    padding_length_by_instruction: dict[str, int] = field(default_factory=dict)
    sub_sentence_present: bool = True
    frozen: bool = True
    force_eval_when_frozen: bool = True
    zero_padded_tokens: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None


@dataclass
class VisionEncoderConfig:
    model_name_or_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    image_size: int = 256
    num_views: int = 2
    position_embedding: str = "view"
    encode_views_separately: bool = True
    frozen: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None
    compute_precision: str = "bf16_autocast"
    position_init_std: float = 0.01
    position_scale_init: float = 0.01
    dropout: float = 0.1


@dataclass
class InteractionConfig:
    hidden_dim: int = 256
    nheads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    enhancer_inner_dim: int = 1024
    text_dropout: float = 0.0
    fusion_dropout: float = 0.0
    fusion_droppath: float = 0.1
    padding_strategy: str = "key_padding_mask"
    residual_style: str = "normalized"
    attention_backend: str = "manual"
    compute_precision: str = "fp32"

    # --- MoT 개선안 플래그 (둘 다 False 이면 원본 동작과 완전히 동일) ---
    # state_as_modality: 로봇 상태를 융합 입력으로 올리고 층마다 상태 전용
    #   어텐션 블록으로 갱신한다. 기존 fusion_layers / text_layers 는 건드리지
    #   않으므로 GroundingDINO 사전학습 초기화가 그대로 유지된다.
    state_as_modality: bool = False
    state_tower_ffn: int = 512
    # layer_tap: 융합 중간층 출력을 액션 디코더 각 층에 나눠 전달한다.
    #   추가 파라미터 없음.
    layer_tap: bool = False
    layer_tap_stride: int = 2

    # vision_tower: 층마다 비전 전용 셀프어텐션 + FFN 을 추가한다.
    #   원본 융합 모듈에는 비전 셀프어텐션도 비전 FFN 도 없어서, 비전 토큰이
    #   6개 층 내내 서로를 보지 못한다. GroundingDINO 체크포인트의
    #   transformer.encoder.layers (7.693M) 가 정확히 이 자리의 가중치인데
    #   TurboVLA 는 로드하지 않는다 — 그것을 회수한다.
    vision_tower: bool = False
    # ffn 2048 은 GroundingDINO linear1(2048,256)/linear2(256,2048) 를 자르지
    # 않고 그대로 복사할 수 있는 유일한 폭이다.
    vision_tower_ffn: int = 2048
    vision_tower_nheads: int = 8
    # vision_tower_coinit=False 로 두면 같은 구조를 랜덤 초기화한다.
    # 논문의 "회수가 핵심인가, 용량 증가가 핵심인가" 대조군이 이 플래그다.
    vision_tower_coinit: bool = True
    # 랜덤 초기화되는 파라미터(Q/K + 게이트)에만 적용할 학습률 배수 (Cosmos 3 방식).
    vision_tower_lr_mult: float = 5.0


@dataclass
class ActionHeadConfig:
    action_dim: int = 7
    state_dim: int = 8
    horizon: int = 12
    num_state_tokens: int = 2
    num_layers: int = 3
    mlp_hidden_dim: int = 512
    state_hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class TurboVLAConfig:
    name: str = "TurboVLA"
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    action: ActionHeadConfig = field(default_factory=ActionHeadConfig)

    def __post_init__(self) -> None:
        if self.name != "TurboVLA":
            raise ValueError(f"model name must be 'TurboVLA', got {self.name!r}")
        if self.vision.num_views < 1:
            raise ValueError("vision.num_views must be positive")
        if self.vision.position_embedding not in {"view", "learned_patch"}:
            raise ValueError("vision.position_embedding must be 'view' or 'learned_patch'")
        if self.vision.compute_precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError("vision.compute_precision must be fp32, bf16, or bf16_autocast")
        if self.text.padding_length is not None:
            if self.text.padding_length < 1:
                raise ValueError("text.padding_length must be positive")
            if self.text.padding_length > self.text.max_length:
                raise ValueError("text.padding_length cannot exceed text.max_length")
        for instruction, length in self.text.padding_length_by_instruction.items():
            if not instruction:
                raise ValueError("text.padding_length_by_instruction cannot contain an empty instruction")
            if length < 1 or length > self.text.max_length:
                raise ValueError(f"invalid text padding length {length} for instruction {instruction!r}")
        if self.interaction.hidden_dim % self.interaction.nheads != 0:
            raise ValueError("interaction.hidden_dim must be divisible by interaction.nheads")
        if self.interaction.padding_strategy not in {"key_padding_mask", "zero_fill"}:
            raise ValueError("interaction.padding_strategy must be key_padding_mask or zero_fill")
        if self.interaction.residual_style not in {"normalized", "pre_norm"}:
            raise ValueError("interaction.residual_style must be normalized or pre_norm")
        if self.interaction.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("interaction.attention_backend must be manual or sdpa")
        if self.interaction.compute_precision not in {"fp32", "bf16_autocast"}:
            raise ValueError("interaction.compute_precision must be fp32 or bf16_autocast")
        if self.interaction.vision_tower:
            if self.interaction.vision_tower_ffn < 1:
                raise ValueError("interaction.vision_tower_ffn must be positive")
            if self.interaction.hidden_dim % self.interaction.vision_tower_nheads != 0:
                raise ValueError("interaction.hidden_dim must be divisible by vision_tower_nheads")
            if self.interaction.vision_tower_lr_mult <= 0:
                raise ValueError("interaction.vision_tower_lr_mult must be positive")
        if self.action.action_dim < 1 or self.action.state_dim < 1 or self.action.horizon < 1:
            raise ValueError("action dimensions and horizon must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TurboVLAConfig":
        data = dict(payload)
        return cls(
            name=str(data.get("name", "TurboVLA")),
            text=TextEncoderConfig(**dict(data.get("text", {}))),
            vision=VisionEncoderConfig(**dict(data.get("vision", {}))),
            interaction=InteractionConfig(**dict(data.get("interaction", {}))),
            action=ActionHeadConfig(**dict(data.get("action", {}))),
        )
