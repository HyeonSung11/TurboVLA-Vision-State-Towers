from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .action_head import TurboVLAActionHead
from .components.fusion import BiAttentionBlock
from .components.transformer import TransformerEncoderLayer
from .components.utils import _get_clones
from .configuration import (
    ActionHeadConfig,
    InteractionConfig,
    TextEncoderConfig,
    TurboVLAConfig,
    VisionEncoderConfig,
)
from .text_encoder import TurboVLATextEncoder
from .vision_encoder import DINOv3VisionEncoder


class VisionProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.output_norm = nn.LayerNorm(out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.skip(tokens) + self.mlp(self.input_norm(tokens)))


class StateTower(nn.Module):
    """로봇 상태 전용 어텐션 블록.

    상태 토큰이 그 시점의 [visual; text] 문맥을 읽고 자신을 갱신한다.
    시각/언어와 파라미터를 공유하지 않는 것이 핵심이다 — 상태는 관절/그리퍼
    같은 소수의 연속값이라 시각 패치와 통계가 달라, 같은 투영 행렬을 쓰면
    서로를 방해한다 (Mixture-of-Transformers, TMLR 2025).

    상태 토큰 수가 2개뿐이라 어텐션 비용은 사실상 무시할 수준이다.
    """

    def __init__(self, hidden_dim: int, nheads: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % nheads:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by nheads {nheads}")
        self.nheads = nheads
        self.head_dim = hidden_dim // nheads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _shape(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        return tensor.view(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)

    def forward(self, state_tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size = state_tokens.shape[0]
        normed = self.norm_attn(state_tokens)
        attention = nn.functional.scaled_dot_product_attention(
            self._shape(self.q_proj(normed), batch_size),
            self._shape(self.k_proj(context), batch_size),
            self._shape(self.v_proj(context), batch_size),
        )
        attention = attention.transpose(1, 2).reshape(batch_size, -1, self.nheads * self.head_dim)
        state_tokens = state_tokens + self.out_proj(attention)
        return state_tokens + self.ffn(self.norm_ffn(state_tokens))


class VisionTower(nn.Module):
    """비전 전용 셀프어텐션 + FFN 블록.

    원본 융합 모듈에는 이 자리가 비어 있다. 비전 토큰은 텍스트로부터의
    크로스 어텐션만 받을 뿐, 512개 패치가 서로를 한 번도 보지 못하고
    비전 전용 FFN 도 없다. 두 뷰 사이의 대응이나 물체-물체 공간 관계를
    백본 이후에 재구성할 통로가 없다는 뜻이다.

    파라미터 이름을 GroundingDINO ``transformer.encoder.layers.*`` 와 맞춰
    두었기 때문에 linear1/linear2/norm1/norm2 는 이름 그대로 복사되고,
    self_attn.value_proj -> in_proj_weight 의 V 구간,
    self_attn.output_proj -> out_proj 로 옮겨 붙는다 (trainer 쪽에서 수행).
    sampling_offsets / attention_weights 는 MSDeformAttn 전용이라 표준
    어텐션의 Q/K 에 대응물이 없다 — 그래서 Q/K 만 랜덤이고, 이것이
    "부분" co-initialization 인 이유다.

    gate 는 0 으로 초기화한다. 그러면 step 0 에서 출력이 입력과 비트 단위로
    같아, 켜는 것만으로 기준선이 퇴행할 여지가 없다.
    """

    def __init__(self, hidden_dim: int, nheads: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % nheads:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by nheads {nheads}")
        self.nheads = nheads
        self.head_dim = hidden_dim // nheads
        self.hidden_dim = hidden_dim
        self.self_attn = nn.MultiheadAttention(hidden_dim, nheads, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # LayerScale 게이트. 기존 fusion_layers 의 gamma_v 와 같은 역할이되
        # 0 초기화라 처음에는 완전한 항등 사상이다.
        self.gamma = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        batch = visual_tokens.shape[0]
        weight = self.self_attn.in_proj_weight
        bias = self.self_attn.in_proj_bias
        query, key, value = nn.functional.linear(visual_tokens, weight, bias).chunk(3, dim=-1)
        shape = lambda t: t.view(batch, -1, self.nheads, self.head_dim).transpose(1, 2)
        # 비전에는 패딩이 없으므로 마스크를 넘기지 않는다. 명시적 마스크를 주면
        # SDPA 고속 경로를 잃고 N^2 어텐션 행렬이 통째로 저장된다.
        attended = nn.functional.scaled_dot_product_attention(shape(query), shape(key), shape(value))
        attended = attended.transpose(1, 2).reshape(batch, -1, self.hidden_dim)
        hidden = self.norm1(visual_tokens + self.self_attn.out_proj(attended))
        hidden = self.norm2(hidden + self.linear2(self.dropout(nn.functional.relu(self.linear1(hidden)))))
        return visual_tokens + self.gamma * (hidden - visual_tokens)


class VisionLanguageInteraction(nn.Module):
    def __init__(self, config: InteractionConfig) -> None:
        super().__init__()
        text_layer = TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=max(1, config.nheads // 2),
            dim_feedforward=config.enhancer_inner_dim,
            dropout=config.text_dropout,
        )
        fusion_layer = BiAttentionBlock(
            v_dim=config.hidden_dim,
            l_dim=config.hidden_dim,
            embed_dim=config.enhancer_inner_dim,
            num_heads=max(1, config.nheads // 2),
            dropout=config.fusion_dropout,
            drop_path=config.fusion_droppath,
            residual_style=config.residual_style,
            attention_backend=config.attention_backend,
        )
        self.text_layers = _get_clones(text_layer, config.num_layers)
        self.fusion_layers = _get_clones(fusion_layer, config.num_layers)
        self.padding_strategy = config.padding_strategy
        self.num_layers = config.num_layers

        # --- 개선안: 원본 경로에는 영향을 주지 않는 추가 모듈 ---
        self.state_as_modality = bool(config.state_as_modality)
        self.layer_tap = bool(config.layer_tap)
        self.layer_tap_stride = max(1, int(config.layer_tap_stride))
        if self.state_as_modality:
            self.state_towers = _get_clones(
                StateTower(
                    hidden_dim=config.hidden_dim,
                    nheads=config.nheads,
                    ffn_dim=config.state_tower_ffn,
                    dropout=config.fusion_dropout,
                ),
                config.num_layers,
            )
        else:
            self.state_towers = None

        self.vision_tower = bool(config.vision_tower)
        if self.vision_tower:
            self.vision_towers = _get_clones(
                VisionTower(
                    hidden_dim=config.hidden_dim,
                    nheads=config.vision_tower_nheads,
                    ffn_dim=config.vision_tower_ffn,
                    dropout=config.fusion_dropout,
                ),
                config.num_layers,
            )
        else:
            self.vision_towers = None

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        text_self_attention_masks: torch.Tensor | None,
        state_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        zero_fill = self.padding_strategy == "zero_fill"
        if zero_fill:
            text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)

        if self.state_as_modality and state_tokens is None:
            raise ValueError("state_as_modality=True requires state_tokens")

        taps: list[torch.Tensor] = []
        for index, (fusion_layer, text_layer) in enumerate(zip(self.fusion_layers, self.text_layers)):
            visual_tokens, text_tokens = fusion_layer(
                v=visual_tokens,
                l=text_tokens,
                attention_mask_v=None,
                attention_mask_l=text_key_padding_mask,
            )
            source_mask = None if text_self_attention_masks is None else ~text_self_attention_masks
            text_tokens = text_layer(
                src=text_tokens.transpose(0, 1),
                src_mask=source_mask,
                src_key_padding_mask=None if zero_fill else text_key_padding_mask,
                pos=None,
            ).transpose(0, 1)
            if zero_fill:
                text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)

            # 텍스트 갱신 뒤, 상태 타워가 읽기 전에 둔다. 그래야 상태가
            # 개선된 시각 표현을 읽는다.
            if self.vision_towers is not None:
                visual_tokens = self.vision_towers[index](visual_tokens)

            if self.state_towers is not None:
                state_tokens = self.state_towers[index](
                    state_tokens, torch.cat([visual_tokens, text_tokens], dim=1)
                )
            if self.layer_tap and (index + 1) % self.layer_tap_stride == 0:
                parts = [visual_tokens, text_tokens]
                if state_tokens is not None:
                    parts.append(state_tokens)
                taps.append(torch.cat(parts, dim=1))
        return visual_tokens, text_tokens, state_tokens, taps


class TurboVLA(nn.Module):
    """Shared TurboVLA architecture for LIBERO and RoboTwin."""

    def __init__(self, config: TurboVLAConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.interaction.hidden_dim
        self.action_dim = int(config.action.action_dim)
        self.chunk_size = int(config.action.horizon)
        self.state_dim = int(config.action.state_dim)
        self.num_views = int(config.vision.num_views)

        self.text_encoder = TurboVLATextEncoder(config.text, hidden_dim=hidden_dim)
        self.vision_encoder = DINOv3VisionEncoder(config.vision)
        self.vision_projection = VisionProjection(
            in_dim=self.vision_encoder.hidden_size,
            out_dim=hidden_dim,
            hidden_dim=max(hidden_dim * 4, self.vision_encoder.hidden_size // 2),
            dropout=config.vision.dropout,
        )

        if config.vision.position_embedding == "learned_patch":
            self.view_embedding = nn.Parameter(torch.zeros(1, self.num_views, 1, hidden_dim))
            self.patch_position_embedding = nn.Parameter(
                torch.zeros(1, self.num_views, self.vision_encoder.num_patches, hidden_dim)
            )
            self.patch_position_scale = nn.Parameter(
                torch.full((1, self.num_views, 1, 1), float(config.vision.position_scale_init))
            )
            nn.init.trunc_normal_(self.patch_position_embedding, std=config.vision.position_init_std)
        else:
            self.view_embedding = nn.Parameter(torch.zeros(1, self.num_views, hidden_dim))
            self.register_parameter("patch_position_embedding", None)
            self.register_parameter("patch_position_scale", None)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)

        self.vision_language_interaction = VisionLanguageInteraction(config.interaction)
        self.action_head = TurboVLAActionHead(
            config=config.action,
            hidden_dim=hidden_dim,
            nheads=config.interaction.nheads,
            dim_feedforward=config.interaction.dim_feedforward,
            state_in_memory=config.interaction.state_as_modality,
        )

    def _normalize_samples(self, samples: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(samples, Mapping):
            if "dinov3" not in samples:
                raise ValueError("samples mapping must contain 'dinov3'")
            pixel_values = samples["dinov3"]
        else:
            pixel_values = samples
        if pixel_values.ndim == 6:
            pixel_values = pixel_values[:, -1]
        if pixel_values.ndim != 5:
            raise ValueError(f"samples must be [B,V,3,H,W] or [B,T,V,3,H,W], got {tuple(pixel_values.shape)}")
        return pixel_values

    def _position_visual_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.config.vision.position_embedding == "learned_patch":
            if tokens.shape[2] != self.patch_position_embedding.shape[2]:
                raise ValueError(
                    f"configured patch position length {self.patch_position_embedding.shape[2]} "
                    f"does not match encoded length {tokens.shape[2]}"
                )
            position = self.patch_position_embedding.to(device=tokens.device, dtype=tokens.dtype)
            scale = self.patch_position_scale.to(device=tokens.device, dtype=tokens.dtype)
            view = self.view_embedding.to(device=tokens.device, dtype=tokens.dtype)
            return tokens + scale * position + view
        view = self.view_embedding[:, :, None, :].to(device=tokens.device, dtype=tokens.dtype)
        return tokens + view

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.vision_encoder(pixel_values)
        tokens = tokens.to(dtype=self.vision_projection.skip.weight.dtype)
        tokens = self.vision_projection(tokens)
        return self._position_visual_tokens(tokens).flatten(1, 2)

    def encode_condition(
        self,
        instructions: Sequence[str],
        samples: torch.Tensor | Mapping[str, torch.Tensor],
        state: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """액션 디코더가 참조할 memory 리스트를 만든다.

        플래그가 모두 꺼져 있으면 원본과 동일하게 [cat(visual, text)] 하나를
        반환한다. layer_tap 이 켜지면 중간층 표현이 여러 개 반환되고,
        state_as_modality 가 켜지면 상태가 융합 입력으로 함께 들어간다.
        """
        pixel_values = self._normalize_samples(samples)
        device = pixel_values.device
        precision_context = nullcontext()
        if self.config.interaction.compute_precision == "bf16_autocast" and device.type == "cuda":
            precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with precision_context:
            text_tokens, text_key_padding_mask, text_self_attention_masks = self.text_encoder(
                instructions,
                device=device,
            )
            if text_tokens.shape[0] != pixel_values.shape[0]:
                raise ValueError("instruction batch size does not match image batch size")
            visual_tokens = self.encode_vision(pixel_values)

            state_tokens = None
            if self.config.interaction.state_as_modality:
                if state is None:
                    raise ValueError("state_as_modality=True requires state in encode_condition")
                # action_head 안의 StateProjection 을 그대로 재사용한다.
                # 모듈을 옮기지 않으므로 체크포인트 키가 유지된다.
                state_tokens = self.action_head.state_projection(
                    state.to(device=visual_tokens.device, dtype=visual_tokens.dtype)
                )

            visual_tokens, text_tokens, state_tokens, taps = self.vision_language_interaction(
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_key_padding_mask=text_key_padding_mask,
                text_self_attention_masks=text_self_attention_masks,
                state_tokens=state_tokens,
            )
            if taps:
                return taps
            parts = [visual_tokens, text_tokens]
            if state_tokens is not None:
                parts.append(state_tokens)
            return [torch.cat(parts, dim=1)]

    def forward(
        self,
        instructions: Sequence[str],
        samples: torch.Tensor | Mapping[str, torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        memories = self.encode_condition(instructions, samples, state)
        action_dtype = self.action_head.decoder.action_queries.weight.dtype
        memories = [memory.to(dtype=action_dtype) for memory in memories]
        return self.action_head(memories, state.to(dtype=action_dtype))

    # Transitional read-only names used only by legacy checkpoint initialization.
    @property
    def dinov3(self):
        return self.vision_encoder.backbone

    @property
    def text_proj(self):
        return self.text_encoder.text_projection

    @property
    def vision_proj(self):
        return self.vision_projection

    @property
    def feature_enhancer(self):
        return self.vision_language_interaction

    @property
    def state_proj(self):
        return self.action_head.state_projection

    @property
    def action_policy(self):
        return self.action_head.decoder


def _arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_turbovla(args: TurboVLAConfig | Mapping[str, Any] | Any) -> TurboVLA:
    if isinstance(args, TurboVLAConfig):
        config = args
    elif isinstance(args, Mapping):
        config = TurboVLAConfig.from_mapping(args)
    else:
        config = TurboVLAConfig(
            text=TextEncoderConfig(
                model_name_or_path=_arg(args, "bert_path", "bert-base-uncased"),
                max_length=int(_arg(args, "max_text_len", 256)),
                padding_length=_arg(args, "text_padding_length", None),
                padding_length_by_instruction=dict(_arg(args, "text_padding_length_by_instruction", {})),
                sub_sentence_present=bool(_arg(args, "sub_sentence_present", True)),
                frozen=bool(_arg(args, "freeze_text_encoder", True)),
                force_eval_when_frozen=True,
                zero_padded_tokens=bool(_arg(args, "zero_padded_text", False)),
                local_files_only=bool(_arg(args, "local_files_only", True)),
                attention_implementation=_arg(args, "text_attention_implementation", None),
            ),
            vision=VisionEncoderConfig(
                model_name_or_path=_arg(args, "dinov3_path", _arg(args, "LOCAL_DINOV3_PATH", "")),
                image_size=int(_arg(args, "image_size", _arg(args, "expected_image_size", 256))),
                num_views=int(_arg(args, "num_views", 2)),
                position_embedding=str(_arg(args, "position_embedding", "view")),
                encode_views_separately=bool(_arg(args, "encode_views_separately", True)),
                frozen=bool(_arg(args, "freeze_vision_encoder", False)),
                local_files_only=bool(_arg(args, "local_files_only", True)),
                attention_implementation=_arg(args, "vision_attention_implementation", None),
                compute_precision=str(_arg(args, "dinov3_precision", "bf16_autocast")),
                dropout=float(_arg(args, "vision_dropout", 0.1)),
            ),
            interaction=InteractionConfig(
                hidden_dim=int(_arg(args, "hidden_dim", 256)),
                nheads=int(_arg(args, "nheads", 8)),
                num_layers=int(_arg(args, "vla_feature_enhancer_layers", 6)),
                dim_feedforward=int(_arg(args, "dim_feedforward", 2048)),
                enhancer_inner_dim=int(_arg(args, "enhancer_inner_dim", 1024)),
                text_dropout=float(_arg(args, "text_dropout", 0.0)),
                fusion_dropout=float(_arg(args, "fusion_dropout", 0.0)),
                fusion_droppath=float(_arg(args, "fusion_droppath", 0.1)),
                padding_strategy=str(_arg(args, "padding_strategy", "key_padding_mask")),
                residual_style=str(_arg(args, "residual_style", "normalized")),
                attention_backend=str(_arg(args, "attention_backend", "manual")),
                compute_precision=str(_arg(args, "interaction_precision", "fp32")),
                state_as_modality=bool(_arg(args, "state_as_modality", False)),
                state_tower_ffn=int(_arg(args, "state_tower_ffn", 512)),
                layer_tap=bool(_arg(args, "layer_tap", False)),
                vision_tower=bool(_arg(args, "vision_tower", False)),
                vision_tower_ffn=int(_arg(args, "vision_tower_ffn", 2048)),
                vision_tower_nheads=int(_arg(args, "vision_tower_nheads", 8)),
                vision_tower_coinit=bool(_arg(args, "vision_tower_coinit", True)),
                vision_tower_lr_mult=float(_arg(args, "vision_tower_lr_mult", 5.0)),
                layer_tap_stride=int(_arg(args, "layer_tap_stride", 2)),
            ),
            action=ActionHeadConfig(
                action_dim=int(_arg(args, "action_dim", 7)),
                state_dim=int(_arg(args, "state_dim", 8)),
                horizon=int(_arg(args, "chunk_size", _arg(args, "action_horizon", 12))),
                num_state_tokens=int(_arg(args, "num_state_tokens", 2)),
                num_layers=int(_arg(args, "act_num_layers", 3)),
                mlp_hidden_dim=int(_arg(args, "act_mlp_hidden_dim", 512)),
                state_hidden_dim=int(_arg(args, "act_state_hidden_dim", 256)),
                dropout=float(_arg(args, "act_dropout", 0.1)),
            ),
        )
    return TurboVLA(config)
