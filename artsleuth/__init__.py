"""
ArtSleuth — Computational Art Analysis Framework.

There's a gap between what connoisseurs *feel* when they look at a
painting and what we can actually quantify.  Brushstroke directionality,
palette warmth, impasto thickness — these are measurable, and yet most
attribution debates still come down to "trust my eye."  This toolkit
tries to put numbers under the intuition, without pretending the
numbers are the whole story.

Powered by DINOv2 and CLIP.  Grounded in Morelli, Berenson, and a
healthy scepticism toward both art historians and algorithms.

Example
-------
>>> import artsleuth
>>> result = artsleuth.analyze("painting.jpg")
>>> print(result.style)
>>> result.explain().save("analysis.png")
"""

from __future__ import annotations

__version__ = "0.2.1"
__author__ = "Danielle Lesin"

from artsleuth.config import AnalysisConfig
from artsleuth.core.attribution import AttributionAnalyzer
from artsleuth.core.brushstroke import BrushstrokeAnalyzer
from artsleuth.core.explainability import ExplainabilityEngine
from artsleuth.core.forgery import ForgeryDetector
from artsleuth.core.style import StyleClassifier
from artsleuth.core.temporal import TemporalRegistry, TemporalStyleModel
from artsleuth.core.workshop import WorkshopDecomposition
from artsleuth.models.fusion import DualBackboneFusion, StyleGuidedAttention

__all__ = [
    "AnalysisConfig",
    "BrushstrokeAnalyzer",
    "StyleClassifier",
    "AttributionAnalyzer",
    "ForgeryDetector",
    "ExplainabilityEngine",
    "TemporalStyleModel",
    "TemporalRegistry",
    "WorkshopDecomposition",
    "DualBackboneFusion",
    "StyleGuidedAttention",
    "analyze",
    "analyse",
]


def analyze(
    image_path: str,
    *,
    config: AnalysisConfig | None = None,
) -> "AnalysisResult":
    """Run the full ArtSleuth analysis pipeline on a single artwork.

    This is the primary entry point for casual usage. For fine-grained
    control over individual analysis stages, instantiate the component
    analyzers directly.

    Parameters
    ----------
    image_path:
        Path to the artwork image (JPEG, PNG, or TIFF).
    config:
        Optional configuration override. When ``None``, sensible defaults
        calibrated for Western easel painting are applied.

    Returns
    -------
    AnalysisResult
        A structured report containing style classification, attribution
        scores, brushstroke features, and an explainability handle for
        generating visual overlays. Forgery screening requires passing a
        reference_artist to run_pipeline directly.
    """
    from artsleuth.core.pipeline import run_pipeline

    return run_pipeline(image_path, config=config or AnalysisConfig())


analyse = analyze

# Deferred import to avoid circular references at module level.
from artsleuth.core.pipeline import AnalysisResult as AnalysisResult  # noqa: E402
