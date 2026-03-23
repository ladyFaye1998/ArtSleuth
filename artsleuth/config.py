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
    speed and memory.  ViT-S and ViT-B/32 are the defaults for rapid
    prototyping; ViT-B/14 and ViT-L/14 are recommended for
    publication-grade results.
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
        Vision-transformer backbone used for feature extraction.
        DINOv2 excels at fine-grained texture; CLIP provides richer
        semantic-stylistic embeddings.
    patch_size:
        Side length (in pixels) of square patches extracted for
        brushstroke analysis.  Smaller patches capture finer strokes
        but increase computation.
    patch_strategy:
        Patch selection strategy.  ``"grid"`` tiles uniformly;
        ``"salient"`` focuses on high-detail regions; ``"adaptive"``
        combines both heuristics.
    confidence_threshold:
        Minimum confidence score (0–1) for a classification or
        attribution result to be considered reportable.
    device:
        PyTorch device string.  ``None`` triggers automatic selection
        (CUDA → MPS → CPU).
    cache_dir:
        Local directory for downloaded model weights.
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
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Dataloader worker count for batch operations.",
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
    use_fusion: bool = Field(
        default=False,
        description=(
            "Use cross-attention backbone fusion instead of concatenation. "
            "Only effective during training; the default inference pipeline "
            "uses feature concatenation regardless of this setting."
        ),
    )
    enable_temporal: bool = Field(
        default=True,
        description="Enable temporal style drift adjustment in attribution.",
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
