"""
Cross-attention fusion of DINOv2 and CLIP backbones.

Standard practice concatenates CLIP and DINOv2 embeddings — treating
texture and semantics as independent channels.  This module instead
uses cross-attention to let CLIP's style-level understanding guide
*where* DINOv2 focuses at the patch level, producing style-aware
structural features that neither backbone achieves alone.

The architecture is deliberately lightweight: a single cross-attention
layer with a learned temperature, followed by an optional residual
connection.  Heavier fusion (e.g. iterative cross-attention towers)
showed diminishing returns in early experiments while increasing
inference cost on high-resolution artwork images.

References
----------
Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
Jose, J. et al. (2025). DINOv2 Meets Text. *CVPR*.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from artsleuth.models.backbones import load_backbone

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# --- Dimension defaults per backbone size -----------------------------------

_SIZE_DIMS: dict[str, tuple[int, int]] = {
    "small": (384, 512),   # DINOv2 ViT-S/14,  CLIP ViT-B/32
    "base":  (768, 768),   # DINOv2 ViT-B/14,  CLIP ViT-L/14
    "large": (1024, 768),  # DINOv2 ViT-L/14,  CLIP ViT-L/14@336px
}

_DINO_DIM = 384
_CLIP_DIM = 512
_FUSED_DIM = _CLIP_DIM


# --- Style-Guided Cross-Attention ------------------------------------------


class StyleGuidedAttention(nn.Module):
    """Cross-attention where CLIP queries guide DINOv2 patch selection.

    CLIP provides the *what* (style/semantics) and DINOv2 provides
    the *where* (texture/structure).  The query comes from CLIP so
    that style-level understanding steers attention across DINOv2's
    spatially rich patch tokens.

    Parameters
    ----------
    dino_dim : int
        Dimensionality of DINOv2 patch embeddings.
    clip_dim : int
        Dimensionality of CLIP embeddings.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability on attention weights.
    """

    def __init__(
        self,
        dino_dim: int = _DINO_DIM,
        clip_dim: int = _CLIP_DIM,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.internal_dim = clip_dim
        self.num_heads = num_heads
        self.head_dim = self.internal_dim // num_heads

        self.q_proj = nn.Linear(clip_dim, self.internal_dim)
        self.k_proj = nn.Linear(dino_dim, self.internal_dim)
        self.v_proj = nn.Linear(dino_dim, self.internal_dim)

        self.temperature = nn.Parameter(torch.ones(1))
        self.attn_dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(self.internal_dim)

    def forward(
        self,
        dino_features: Tensor,
        clip_features: Tensor,
    ) -> Tensor:
        """Fuse DINOv2 patches with CLIP semantic guidance.

        Parameters
        ----------
        dino_features : Tensor
            DINOv2 embeddings, ``(B, N, dino_dim)`` or
            ``(B, dino_dim)`` for CLS-only mode.
        clip_features : Tensor
            CLIP embeddings, ``(B, clip_dim)``.

        Returns
        -------
        Tensor
            Fused features, ``(B, fused_dim)``.
        """
        if dino_features.ndim == 2:
            dino_features = dino_features.unsqueeze(1)

        clip_features = clip_features.unsqueeze(1)  # (B, 1, clip_dim)

        B, N, _ = dino_features.shape
        H, d = self.num_heads, self.head_dim

        Q = self.q_proj(clip_features).view(B, 1, H, d).transpose(1, 2)
        K = self.k_proj(dino_features).view(B, N, H, d).transpose(1, 2)
        V = self.v_proj(dino_features).view(B, N, H, d).transpose(1, 2)

        scale = math.sqrt(d) * self.temperature
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, 1, d)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, self.internal_dim)
        )
        return self.out_norm(out)


# --- Dual-Backbone Fusion --------------------------------------------------


class DualBackboneFusion(nn.Module):
    """Orchestrates DINOv2 + CLIP extraction and cross-attention fusion.

    Both backbones are lazy-loaded on the first forward pass so the
    module can be constructed cheaply and serialised without embedding
    multi-GB checkpoint tensors.

    Parameters
    ----------
    device : str
        PyTorch device string.
    cache_dir : Path | None
        Local directory for downloaded backbone weights.
    backbone_size : str
        One of ``"small"``, ``"base"``, or ``"large"``.
    residual : bool
        When ``True``, the original CLIP embedding is concatenated
        with the cross-attention output and projected back to
        ``fused_dim``, giving the network a direct semantic shortcut.
    """

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: Path | None = None,
        *,
        backbone_size: str = "small",
        residual: bool = True,
    ) -> None:
        super().__init__()
        self._device = device
        self._cache_dir = cache_dir
        self._backbone_size = backbone_size
        self.residual = residual

        dino_dim, clip_dim = _SIZE_DIMS.get(
            backbone_size, (_DINO_DIM, _CLIP_DIM),
        )
        self._fused_dim = clip_dim

        self._dino: nn.Module | None = None
        self._clip: nn.Module | None = None

        self.attention = StyleGuidedAttention(
            dino_dim=dino_dim, clip_dim=clip_dim,
        )

        if residual:
            self.residual_proj = nn.Linear(
                clip_dim + clip_dim, clip_dim,
            )

    # --- Lazy backbone loading ----------------------------------------------

    def _ensure_backbones(self) -> None:
        """Load backbones on first use."""
        if self._dino is not None:
            return

        from artsleuth.config import BackboneSize, BackboneType

        size = BackboneSize(self._backbone_size)
        logger.info(
            "Lazy-loading fusion backbones (size=%s).", self._backbone_size,
        )
        self._dino = load_backbone(
            BackboneType.DINO_V2,
            size=size,
            device=self._device,
            cache_dir=self._cache_dir,
        )
        self._clip = load_backbone(
            BackboneType.CLIP,
            size=size,
            device=self._device,
            cache_dir=self._cache_dir,
        )

    @torch.no_grad()
    def _extract(
        self, image_tensor: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run both backbones and return raw features."""
        assert self._dino is not None and self._clip is not None
        dino_out = self._dino(image_tensor)
        clip_out = self._clip(image_tensor)
        return dino_out, clip_out

    # --- Forward ------------------------------------------------------------

    def forward(self, image_tensor: Tensor) -> Tensor:
        """Extract and fuse features from a batch of images.

        Parameters
        ----------
        image_tensor : Tensor
            Pre-processed images, ``(B, 3, H, W)``.

        Returns
        -------
        Tensor
            Fused embedding, ``(B, output_dim)``.
        """
        self._ensure_backbones()
        dino_features, clip_features = self._extract(image_tensor)
        fused = self.attention(dino_features, clip_features)

        if self.residual:
            fused = torch.cat([fused, clip_features], dim=-1)
            fused = self.residual_proj(fused)

        return fused

    @property
    def output_dim(self) -> int:
        """Dimensionality of the fused output embedding."""
        return self._fused_dim


# --- Utility ----------------------------------------------------------------


def fusion_output_dim(backbone_size: str = "small") -> int:
    """Return the output dimensionality of the fusion module."""
    _, clip_dim = _SIZE_DIMS.get(backbone_size, (_DINO_DIM, _CLIP_DIM))
    return clip_dim
