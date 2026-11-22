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
        from mribreast.models.networks import make_network

        if int(embed_dim) < 1:
            raise ValueError(f"embed_dim must be a positive integer, got {embed_dim}")
        self.model_name = str(model_name)
        self.embed_dim = int(embed_dim)
        common = dict(spatial_dims=spatial_dims, in_channels=in_channels, num_classes=self.embed_dim)
        self.global_encoder = make_network(self.model_name, extra=dict(extra or {}), **common)
        self.local_encoder = make_network(self.model_name, extra=dict(extra or {}), **common)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(float(dropout)),
                                  nn.Linear(2 * self.embed_dim, int(num_classes)))

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if not isinstance(x, (tuple, list)) or len(x) != 2 or not all(isinstance(v, torch.Tensor) for v in x):
            what = type(x).__name__ + (f" of length {len(x)}" if isinstance(x, (tuple, list)) else "")
            raise TypeError(f"DualStreamNetwork expects a (global, local) tuple or list of two tensors, got {what}")
        global_x, local_x = x
        if global_x.shape[0] != local_x.shape[0]:
            raise ValueError(f"global and local batch sizes differ: {global_x.shape[0]} vs {local_x.shape[0]}")
        g = self.global_encoder(global_x).reshape(global_x.shape[0], -1)
        loc = self.local_encoder(local_x).reshape(local_x.shape[0], -1)
        return self.head(torch.cat([g, loc], dim=1))


__all__ = ["DualStreamNetwork", "DUAL_STREAM_DROPOUT"]
