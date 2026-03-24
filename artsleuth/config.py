"""
Configuration management for ArtSleuth analysis pipelines.

Single validated configuration object governing every stage of the
analysis.  Defaults are tuned for Western easel painting (oil on canvas,
roughly 15th–19th century) since that is the best-represented category
in current training data, but all parameters are overridable for other
traditions (ink wash, miniature painting, etc.).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class BackboneType(str, Enum):
    """Supported vision-transformer backbone architectures."""

    DINO_V2 = "dinov2"
    CLIP = "clip"


class BackboneSize(str, Enum):
    """Model-size variant for each backbone family.

    Larger variants produce richer features at the cost of inference
    speed and memory.  ViT-B/14 and ViT-L/14 are the defaults, matching
    shipped pretrained weights.  ViT-S/14 and ViT-B/32 (``small``) trade
    accuracy for speed.
    """

    SMALL = "small"
    BASE = "base"
    LARGE = "large"


class PatchStrategy(str, Enum):
    """Strategy for extracting analysis patches from an artwork image."""

    GRID = "grid"
    SALIENT = "salient"
    ADAPTIVE = "adaptive"


class AnalysisConfig(BaseModel):
    """Master configuration for an ArtSleuth analysis run.

    Attributes
    ----------
    backbone:
        Vision-transformer backbone for feature extraction.
    backbone_size:
        Model-size variant (small / base / large).  ``base`` matches
        shipped pretrained weights.
    patch_size:
        Side length (pixels) of analysis patches.
    patch_strategy:
        Patch selection strategy (grid / salient / adaptive).
    confidence_threshold:
        Minimum confidence (0–1) for reportable results.
    device:
        PyTorch device string.  ``None`` → auto-detect.
    cache_dir:
        Local directory for downloaded model weights.
    max_resolution:
        Maximum image side length before downscaling.
    enable_art_preprocessing:
        Apply optional art-specific transforms (experimental).
    enable_temporal:
        Attempt temporal style drift estimation (requires user data).
    enable_workshop:
        Use Bayesian workshop decomposition.
    workshop_max_hands:
        Upper bound on inferred workshop hands.
    """

    backbone: BackboneType = Field(
        default=BackboneType.DINO_V2,
        description="Vision-transformer backbone for feature extraction.",
    )
    patch_size: int = Field(
        default=224,
        ge=64,
        le=1024,
        description="Side length in pixels of analysis patches.",
    )
    patch_strategy: PatchStrategy = Field(
        default=PatchStrategy.ADAPTIVE,
        description="Patch selection strategy.",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for reportable results.",
    )
    device: Optional[str] = Field(
        default=None,
        description="PyTorch device string; None for auto-detection.",
    )
    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "artsleuth",
        description="Directory for downloaded model weights.",
    )
    max_resolution: int = Field(
        default=2048,
        ge=512,
        le=8192,
        description="Maximum side length before downscaling input image.",
    )
    enable_art_preprocessing: bool = Field(
        default=False,
        description=(
            "Apply art-specific preprocessing (varnish correction, "
            "craquelure suppression, canvas texture filtering) before "
            "backbone encoding. Experimental — not used in published benchmarks."
        ),
    )
    # --- Novel module settings -----------------------------------------------

    backbone_size: BackboneSize = Field(
        default=BackboneSize.BASE,
        description=(
            "Model-size variant.  'small' → DINOv2 ViT-S/14 + CLIP ViT-B/32 "
            "(fast, 384+512 dim).  'base' → DINOv2 ViT-B/14 + CLIP ViT-L/14 "
            "(recommended, 768+768 dim, matches pretrained weights).  'large' → "
            "DINOv2 ViT-L/14 + CLIP ViT-L/14@336px (highest quality, needs "
            "≥24 GB VRAM)."
        ),
    )
    enable_temporal: bool = Field(
        default=False,
        description=(
            "Attempt temporal style drift estimation (requires a populated "
            "TemporalRegistry; no bundled data is shipped, so this has no "
            "effect out of the box)."
        ),
    )
    enable_workshop: bool = Field(
        default=True,
        description=(
            "Use Bayesian workshop decomposition instead of flat k-means "
            "for brushstroke clustering."
        ),
    )
    workshop_max_hands: int = Field(
        default=6,
        ge=2,
        le=12,
        description="Upper bound on number of workshop hands to infer.",
    )

    model_config = {"frozen": True}

    def resolve_device(self) -> str:
        """Return an explicit device string, auto-detecting when necessary."""
        if self.device is not None:
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
