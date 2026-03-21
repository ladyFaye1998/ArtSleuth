"""
ArtSleuth — AI Art Forensics & Analysis Framework.

A computational toolkit for brushstroke analysis, style attribution,
forgery detection, and interpretable visual explanations in fine art,
powered by vision transformers and grounded in art-historical methodology.

Developed at the intersection of computer science and art history,
ArtSleuth bridges quantitative machine-learning techniques with the
qualitative reasoning that has guided connoisseurship for centuries.

Example
-------
>>> import artsleuth
>>> result = artsleuth.analyze("painting.jpg")
>>> print(result.style)
>>> result.explain().save("analysis.png")
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Danielle Lesin"

from artsleuth.config import AnalysisConfig
from artsleuth.core.attribution import AttributionAnalyzer
from artsleuth.core.brushstroke import BrushstrokeAnalyzer
from artsleuth.core.explainability import ExplainabilityEngine
from artsleuth.core.forgery import ForgeryDetector
from artsleuth.core.style import StyleClassifier

__all__ = [
    "AnalysisConfig",
    "BrushstrokeAnalyzer",
    "StyleClassifier",
    "AttributionAnalyzer",
    "ForgeryDetector",
    "ExplainabilityEngine",
    "analyze",
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
        scores, brushstroke features, forgery indicators, and an
        explainability handle for generating visual overlays.
    """
    from artsleuth.core.pipeline import run_pipeline

    return run_pipeline(image_path, config=config or AnalysisConfig())


# Deferred import to avoid circular references at module level.
from artsleuth.core.pipeline import AnalysisResult as AnalysisResult  # noqa: E402
