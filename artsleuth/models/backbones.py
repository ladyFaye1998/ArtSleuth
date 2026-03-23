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
everything works offline once downloaded.  Model size is configurable
via :class:`~artsleuth.config.BackboneSize`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from artsleuth.config import BackboneSize, BackboneType

logger = logging.getLogger(__name__)

# --- Size → model-name mappings ---------------------------------------------

_DINO_MODELS: dict[str, str] = {
    "small": "dinov2_vits14",
    "base": "dinov2_vitb14",
    "large": "dinov2_vitl14",
}

_CLIP_MODELS: dict[str, str] = {
    "small": "ViT-B/32",
    "base": "ViT-L/14",
    "large": "ViT-L/14@336px",
}

_DINO_HF_FALLBACK: dict[str, str] = {
    "small": "facebook/dinov2-small",
    "base": "facebook/dinov2-base",
    "large": "facebook/dinov2-large",
}

_CLIP_HF_FALLBACK: dict[str, str] = {
    "small": "openai/clip-vit-base-patch32",
    "base": "openai/clip-vit-large-patch14",
    "large": "openai/clip-vit-large-patch14-336",
}

_DINO_DIMS: dict[str, int] = {"small": 384, "base": 768, "large": 1024}
_CLIP_DIMS: dict[str, int] = {"small": 512, "base": 768, "large": 768}

_BACKBONE_CACHE: dict[str, nn.Module] = {}


# --- Public API -------------------------------------------------------------


def load_backbone(
    backbone_type: "BackboneType",
    *,
    size: "BackboneSize | None" = None,
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
    size:
        Model-size variant (small / base / large).  Defaults to
        ``BackboneSize.SMALL`` for backward compatibility.
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
    from artsleuth.config import BackboneSize, BackboneType

    if size is None:
        size = BackboneSize.BASE
    size_key = size.value

    cache_key = f"{backbone_type.value}_{size_key}_{device}"
    if cache_key in _BACKBONE_CACHE:
        return _BACKBONE_CACHE[cache_key]

    if backbone_type == BackboneType.DINO_V2:
        model = _load_dinov2(size_key, device, cache_dir)
    elif backbone_type == BackboneType.CLIP:
        model = _load_clip(size_key, device, cache_dir)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_type}")

    _BACKBONE_CACHE[cache_key] = model
    return model


def embedding_dim(
    backbone_type: "BackboneType",
    size: "BackboneSize | None" = None,
) -> int:
    """Return the output embedding dimensionality for a backbone."""
    from artsleuth.config import BackboneSize, BackboneType

    if size is None:
        size = BackboneSize.BASE
    key = size.value

    if backbone_type == BackboneType.DINO_V2:
        return _DINO_DIMS[key]
    if backbone_type == BackboneType.CLIP:
        return _CLIP_DIMS[key]
    raise ValueError(f"Unsupported backbone: {backbone_type}")


# --- Wrapper modules --------------------------------------------------------


class _DINOv2Wrapper(nn.Module):
    """Wraps the DINOv2 model to return CLS-token embeddings."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        if isinstance(features, dict):
            return features.get(
                "x_norm_clstoken",
                features.get("cls_token", features),
            )
        return features


class _CLIPVisualWrapper(nn.Module):
    """Wraps the CLIP visual encoder to match the expected interface."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(x).float()


# --- Loader implementations ------------------------------------------------


def _load_dinov2(
    size_key: str, device: str, cache_dir: Path | None,
) -> nn.Module:
    hub_name = _DINO_MODELS[size_key]
    logger.info("Loading DINOv2 backbone (%s)…", hub_name)
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2", hub_name, pretrained=True,
        )
    except Exception:
        hf_name = _DINO_HF_FALLBACK[size_key]
        logger.warning(
            "torch.hub load failed; falling back to HuggingFace (%s).",
            hf_name,
        )
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            hf_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    wrapper = _DINOv2Wrapper(model)
    wrapper.eval()
    return wrapper.to(device)


def _load_clip(
    size_key: str, device: str, cache_dir: Path | None,
) -> nn.Module:
    clip_name = _CLIP_MODELS[size_key]
    logger.info("Loading CLIP backbone (%s)…", clip_name)
    try:
        import clip

        model, _ = clip.load(clip_name, device=device)
        wrapper = _CLIPVisualWrapper(model)
        wrapper.eval()
        return wrapper
    except ImportError:
        hf_name = _CLIP_HF_FALLBACK[size_key]
        logger.info(
            "openai-clip not installed; using HuggingFace (%s).", hf_name,
        )
        from transformers import CLIPModel

        model = CLIPModel.from_pretrained(
            hf_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        model = model.vision_model
        model.eval()
        return model.to(device)
