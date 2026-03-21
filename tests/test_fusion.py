"""Tests for artsleuth.models.fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from artsleuth.models.fusion import (
    DualBackboneFusion,
    StyleGuidedAttention,
    fusion_output_dim,
)


class TestStyleGuidedAttention:
    def test_shape(self) -> None:
        attn = StyleGuidedAttention(dino_dim=384, clip_dim=512)
        dino = torch.randn(2, 16, 384)
        clip = torch.randn(2, 512)
        out = attn(dino, clip)
        assert out.shape == (2, 512)

    def test_2d_input(self) -> None:
        attn = StyleGuidedAttention(dino_dim=384, clip_dim=512)
        dino = torch.randn(2, 384)
        clip = torch.randn(2, 512)
        out = attn(dino, clip)
        assert out.shape == (2, 512)


def test_fusion_output_dim() -> None:
    assert fusion_output_dim() == 512


def test_dual_backbone_fusion_init() -> None:
    model = DualBackboneFusion()
    assert isinstance(model, nn.Module)
