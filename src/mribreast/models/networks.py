"""Image networks for 3D volumes and 2D images (``fcn``, ``densenet121``, ``resnet10``, ``resnet18``)."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

MRI_MODELS = ("fcn", "densenet121", "resnet10", "resnet18")


class SimpleFCN(nn.Module):
    """Port of legacy ``Simple3DFCN`` for 2D or 3D input.

    3x (conv k3 p1 -> ReLU -> maxpool 2) with widths 8/16/32, a 1x1 conv to
    ``num_classes`` and global average pooling (no Linear layers).
    Difference: the legacy class hard-coded ``in_channels=1`` although the
    fusion workflow passed ``in_channels`` (TypeError); here it is a parameter.
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 1, spatial_dims: int = 3):
        super().__init__()
        conv = {2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]
        pool = {2: nn.MaxPool2d, 3: nn.MaxPool3d}[spatial_dims]
        gap = {2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}[spatial_dims]
        self.conv1 = conv(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = conv(8, 16, kernel_size=3, padding=1)
        self.conv3 = conv(16, 32, kernel_size=3, padding=1)
        self.pool = pool(2)
        self.classifier_conv = conv(32, num_classes, kernel_size=1)
        self.global_pool = gap(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.classifier_conv(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)


class Simple3DFCN(SimpleFCN):
    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        super().__init__(num_classes=num_classes, in_channels=in_channels, spatial_dims=3)


def make_network(model_name: str, *, spatial_dims: int, in_channels: int, num_classes: int,
                 extra: Mapping[str, Any] | None = None) -> nn.Module:
    """Legacy ``make_mri_network`` generalised to ``spatial_dims`` 2 or 3 (``mri_extra_json`` -> ``extra``)."""
    extra = dict(extra or {})
    if spatial_dims not in (2, 3):
        raise ValueError("spatial_dims must be 2 or 3")
    if model_name == "fcn":
        return SimpleFCN(num_classes=num_classes, in_channels=in_channels, spatial_dims=spatial_dims, **extra)
    if model_name == "densenet121":
        from monai.networks.nets import DenseNet121

        return DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=num_classes, **extra)
    if model_name in ("resnet10", "resnet18"):
        from monai.networks.nets import resnet10, resnet18

        constructor = resnet10 if model_name == "resnet10" else resnet18
        return constructor(spatial_dims=spatial_dims, n_input_channels=in_channels, num_classes=num_classes, **extra)
    raise ValueError(f"unknown MRI model {model_name!r}; expected one of {MRI_MODELS}")
