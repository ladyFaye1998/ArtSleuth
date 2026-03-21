"""
Configuration management for ArtSleuth analysis pipelines.

Provides a single, validated configuration object that governs every
stage of the analysis — from preprocessing parameters to model selection
and output formatting.  Sensible defaults are calibrated for Western
easel painting (oil on canvas, 15th–19th century), but all values are
user-overridable for broader applicability.
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
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Dataloader worker count for batch operations.",
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
