"""Dual-stream network for ``box_mode=dual_stream``: global volume + local box crop.

Two INDEPENDENT encoders (no weight sharing), each ``make_network(name, ..., num_classes=embed_dim)``
so every backbone of :data:`mribreast.models.networks.MRI_MODELS` works unchanged; their embeddings are
concatenated, then ``ReLU -> Dropout(dropout) -> Linear(2 * embed_dim, num_classes)``.

Input: a ``(global [B, C, *target_size], local [B, C, *box_local_size])`` tuple (a list is accepted).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

#: Dropout before the fused head (no config field; recorded in ``image_branch.json`` -> network.dual_stream).
DUAL_STREAM_DROPOUT = 0.2


class DualStreamNetwork(nn.Module):
    def __init__(self, model_name: str, *, spatial_dims: int, in_channels: int, num_classes: int,
                 embed_dim: int = 128, dropout: float = DUAL_STREAM_DROPOUT,
                 extra: Mapping[str, Any] | None = None):
        super().__init__()
        raise NotImplementedError("DualStreamNetwork is not implemented yet")

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:  # pragma: no cover - stub
        raise NotImplementedError("DualStreamNetwork is not implemented yet")


__all__ = ["DualStreamNetwork", "DUAL_STREAM_DROPOUT"]
