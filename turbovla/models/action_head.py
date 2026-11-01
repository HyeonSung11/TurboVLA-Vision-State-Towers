from __future__ import annotations

import torch
from torch import nn

from .components.utils import MLP
from .configuration import ActionHeadConfig


class StateProjection(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int) -> None:
        super().__init__()
        self.num_tokens = int(config.num_state_tokens)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_hidden_dim, self.num_tokens * self.hidden_dim),
        )
        self.position = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_dim) * 0.02)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2:
            raise ValueError(f"state must be [B,D] or [B,T,D], got {tuple(state.shape)}")
        tokens = self.net(state).view(state.shape[0], self.num_tokens, self.hidden_dim)
        return self.output_norm(tokens + self.position.to(device=tokens.device, dtype=tokens.dtype))


class ACTDecoder(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int) -> None:
        super().__init__()
        self.horizon = int(config.horizon)
        self.action_queries = nn.Embedding(self.horizon, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.action_projection = MLP(hidden_dim, config.mlp_hidden_dim, config.action_dim, 3)

    def forward(self, memory: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """memory 가 텐서 하나면 원본과 동일하게 동작한다.

        리스트가 들어오면 디코더 층 k 가 memory[k] 를 참조한다 (layer_tap).
        리스트 길이가 층 수보다 짧으면 마지막 것을 재사용한다.
        """
        if not isinstance(memory, (list, tuple)):
            queries = self.action_queries.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
            hidden = self.decoder(tgt=queries, memory=memory)
            return torch.tanh(self.action_projection(hidden))

        memories = list(memory)
        hidden = self.action_queries.weight.unsqueeze(0).expand(memories[0].shape[0], -1, -1)
        for index, layer in enumerate(self.decoder.layers):
            hidden = layer(hidden, memories[min(index, len(memories) - 1)])
        if self.decoder.norm is not None:
            hidden = self.decoder.norm(hidden)
        return torch.tanh(self.action_projection(hidden))


class TurboVLAActionHead(nn.Module):
    def __init__(
        self,
        config: ActionHeadConfig,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        state_in_memory: bool = False,
    ) -> None:
        super().__init__()
        self.state_projection = StateProjection(config, hidden_dim)
        self.decoder = ACTDecoder(config, hidden_dim, nheads, dim_feedforward)
        # state_as_modality 가 켜지면 상태가 이미 융합 단계에서 memory 에
        # 들어가 있으므로 여기서 다시 붙이지 않는다.
        self.state_in_memory = bool(state_in_memory)

    def forward(
        self,
        vision_language_tokens: torch.Tensor | list[torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(vision_language_tokens, (list, tuple)):
            memories = [vision_language_tokens]
            single = True
        else:
            memories = list(vision_language_tokens)
            single = False

        if not self.state_in_memory:
            state = state.to(device=memories[0].device, dtype=memories[0].dtype)
            state_tokens = self.state_projection(state)
            memories = [torch.cat([memory, state_tokens], dim=1) for memory in memories]

        return self.decoder(memories[0] if single else memories)
