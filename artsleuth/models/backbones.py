"""
Vision-transformer backbone loading and management.

Two backbones, two philosophies:

  * **DINOv2** — pure vision, no language.  Sees texture, brushstroke
    direction, impasto thickness — all the physical stuff that Morelli
    cared about.  This is your low-level "hand of the artist" backbone.

  * **CLIP** — vision *plus* language.  Knows that "Baroque" is a thing
    before it ever sees a Caravaggio.  Perfect for style classification,
    where the categories are as much cultural agreement as visual reality.

Both load from HuggingFace Hub or torch.hub, cached locally so
everything works offline once downloaded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from artsleuth.config import BackboneType

logger = logging.getLogger(__name__)

# --- Constants --------------------------------------------------------------

_DINO_V2_MODEL = "dinov2_vits14"
_CLIP_MODEL = "ViT-B/32"

_BACKBONE_CACHE: dict[str, nn.Module] = {}


# --- Public API -------------------------------------------------------------


def load_backbone(
    backbone_type: "BackboneType",
    *,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> nn.Module:
    """Load a pre-trained vision-transformer backbone.

    Loaded models are cached in-process to avoid redundant weight
    materialisation across analysis stages.

    Parameters
    ----------
    backbone_type:
        Backbone architecture to load.
    device:
        PyTorch device string.
    cache_dir:
        Local directory for downloaded weights.

    Returns
    -------
    nn.Module
        A feature-extraction model that accepts a batch of images
        ``(B, 3, H, W)`` and returns embeddings ``(B, D)``.
    """
    from artsleuth.config import BackboneType

    cache_key = f"{backbone_type.value}_{device}"
    if cache_key in _BACKBONE_CACHE:
        return _BACKBONE_CACHE[cache_key]

    if backbone_type == BackboneType.DINO_V2:
        model = _load_dinov2(device, cache_dir)
    elif backbone_type == BackboneType.CLIP:
        model = _load_clip(device, cache_dir)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_type}")

    _BACKBONE_CACHE[cache_key] = model
    return model


def embedding_dim(backbone_type: "BackboneType") -> int:
    """Return the output embedding dimensionality for a backbone."""
    from artsleuth.config import BackboneType

    dims = {
        BackboneType.DINO_V2: 384,   # ViT-S/14
        BackboneType.CLIP: 512,       # ViT-B/32
    }
    return dims[backbone_type]


# --- Loader Implementations ------------------------------------------------


class _DINOv2Wrapper(nn.Module):
    """Wraps the DINOv2 model to return CLS-token embeddings."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        if isinstance(features, dict):
            return features.get("x_norm_clstoken", features.get("cls_token", features))
        return features


class _CLIPVisualWrapper(nn.Module):
    """Wraps the CLIP visual encoder to match the expected interface."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(x).float()


def _load_dinov2(device: str, cache_dir: Path | None) -> nn.Module:
    """Load DINOv2 ViT-S/14 from torch.hub."""
    logger.info("Loading DINOv2 backbone (%s)…", _DINO_V2_MODEL)
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2",
            _DINO_V2_MODEL,
            pretrained=True,
        )
    except Exception:
        logger.warning(
            "torch.hub load failed; attempting HuggingFace fallback."
        )
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            "facebook/dinov2-small",
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    wrapper = _DINOv2Wrapper(model)
    wrapper.eval()
    return wrapper.to(device)


def _load_clip(device: str, cache_dir: Path | None) -> nn.Module:
    """Load CLIP ViT-B/32."""
    logger.info("Loading CLIP backbone (%s)…", _CLIP_MODEL)
    try:
        import clip

        model, _ = clip.load(_CLIP_MODEL, device=device)
        wrapper = _CLIPVisualWrapper(model)
        wrapper.eval()
        return wrapper
    except ImportError:
        logger.info("openai-clip not installed; using HuggingFace transformers.")
        from transformers import CLIPModel

        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        model = model.vision_model
        model.eval()
        return model.to(device)
