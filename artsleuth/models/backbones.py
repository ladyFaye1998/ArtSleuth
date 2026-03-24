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
    backbone_type: BackboneType,
    *,
    size: BackboneSize | None = None,
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
        ``BackboneSize.BASE`` to match shipped pretrained weights.
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
    backbone_type: BackboneType,
    size: BackboneSize | None = None,
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
    """Wraps DINOv2 to return CLS-token embeddings as a plain tensor.

    Handles both torch.hub (returns dict or tensor) and HuggingFace
    transformers (returns BaseModelOutputWithPooling) output formats.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "last_hidden_state"):
            return features.last_hidden_state[:, 0, :]
        if isinstance(features, dict):
            for key in ("x_norm_clstoken", "cls_token"):
                val = features.get(key)
                if val is not None and isinstance(val, torch.Tensor):
                    return val
        raise TypeError(f"Unexpected DINOv2 output: {type(features)}")


class _CLIPVisualWrapper(nn.Module):
    """Wraps the OpenAI ``clip`` package model to return image features."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(x).float()


class _CLIPHFWrapper(nn.Module):
    """Wraps a HuggingFace ``CLIPModel`` to return projected image features.

    Retains the full model so the text encoder remains available for
    zero-shot classification.
    """

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vision_out = self.clip_model.vision_model(pixel_values=x)
        pooled = vision_out[1] if not isinstance(vision_out, torch.Tensor) else vision_out
        projected = self.clip_model.visual_projection(pooled)
        return projected.float()

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return L2-normalised text embeddings via the CLIP text encoder.

        Manually runs text_model + text_projection to avoid
        inconsistencies with ``get_text_features`` across transformers
        versions.
        """
        text_out = self.clip_model.text_model(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        pooled = text_out[1] if not isinstance(text_out, torch.Tensor) else text_out
        projected = self.clip_model.text_projection(pooled)
        projected = projected.float()
        return projected / (projected.norm(dim=-1, keepdim=True) + 1e-12)


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
    except Exception:
        hf_name = _CLIP_HF_FALLBACK[size_key]
        logger.info(
            "openai-clip unavailable; using HuggingFace (%s).", hf_name,
        )
        from transformers import CLIPModel

        model = CLIPModel.from_pretrained(
            hf_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        wrapper = _CLIPHFWrapper(model)
        wrapper.eval()
        return wrapper.to(device)
