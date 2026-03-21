"""
Unified analysis pipeline.

Orchestrates the full analysis sequence: preprocessing → brushstrokes →
style → attribution → forgery → explainability, all in one call.  For
finer control over individual stages, use the component modules directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from artsleuth.config import AnalysisConfig
    from artsleuth.core.attribution import AttributionReport
    from artsleuth.core.brushstroke import BrushstrokeReport
    from artsleuth.core.explainability import ExplanationMap
    from artsleuth.core.forgery import ForgeryReport
    from artsleuth.core.style import StyleReport


@dataclass
class AnalysisResult:
    """Complete ArtSleuth analysis for a single artwork.

    Attributes
    ----------
    image_path:
        Path to the analysed image file.
    style:
        Style classification (period, school, technique).
    brushstrokes:
        Brushstroke pattern analysis.
    attribution:
        Artist/workshop attribution ranking.
    forgery:
        Forgery anomaly screening (``None`` if no reference artist
        was specified).
    """

    image_path: str
    style: "StyleReport"
    brushstrokes: "BrushstrokeReport"
    attribution: "AttributionReport"
    forgery: "ForgeryReport | None" = None

    def explain(self, target: str = "attribution") -> "ExplanationMap":
        """Generate an interpretable visual overlay for the analysis.

        Parameters
        ----------
        target:
            Which verdict to explain (``"attribution"``, ``"style"``,
            or ``"forgery"``).

        Returns
        -------
        ExplanationMap
            Heatmap overlay highlighting the most salient image regions.
        """
        from artsleuth.config import AnalysisConfig
        from artsleuth.core.explainability import ExplainabilityEngine

        image = Image.open(self.image_path).convert("RGB")
        engine = ExplainabilityEngine(AnalysisConfig())
        return engine.gradcam(image, target_label=target)

    def summary(self) -> str:
        """Return a concise human-readable summary of the analysis."""
        lines = [
            f"ArtSleuth Analysis — {self.image_path}",
            f"{'─' * 50}",
            f"  Period     : {self.style.period.label} ({self.style.period.confidence:.0%})",
            f"  School     : {self.style.school.label} ({self.style.school.confidence:.0%})",
            f"  Technique  : {self.style.technique.label} ({self.style.technique.confidence:.0%})",
            f"  Attribution: {self.attribution.consensus_artist} "
            f"({self.attribution.consensus_confidence:.0%})",
        ]
        if self.attribution.multi_hand_flag:
            lines.append("  ⚑ Multiple hands detected (possible workshop production).")
        if self.forgery and self.forgery.is_flagged:
            lines.append(
                f"  ⚑ Anomaly flag raised (score {self.forgery.anomaly_score:.2f}) — "
                "further technical examination recommended."
            )
        return "\n".join(lines)


def run_pipeline(
    image_path: str,
    *,
    config: "AnalysisConfig",
    reference_artist: str | None = None,
) -> AnalysisResult:
    """Execute the full analysis pipeline.

    Parameters
    ----------
    image_path:
        Path to the artwork image.
    config:
        Analysis configuration.
    reference_artist:
        If provided, the painting is additionally screened for
        forgery anomalies against this artist's reference corpus.

    Returns
    -------
    AnalysisResult
        Structured analysis report.
    """
    from artsleuth.core.attribution import AttributionAnalyzer
    from artsleuth.core.brushstroke import BrushstrokeAnalyzer
    from artsleuth.core.forgery import ForgeryDetector
    from artsleuth.core.style import StyleClassifier

    image = Image.open(image_path).convert("RGB")

    brushstroke_analyzer = BrushstrokeAnalyzer(config)
    style_classifier = StyleClassifier(config)
    attribution_analyzer = AttributionAnalyzer(config)

    brushstroke_report = brushstroke_analyzer.analyze(image)
    style_report = style_classifier.classify(image)
    attribution_report = attribution_analyzer.attribute(
        image,
        brushstroke_report=brushstroke_report,
        style_report=style_report,
    )

    forgery_report = None
    if reference_artist is not None:
        detector = ForgeryDetector(config)
        forgery_report = detector.detect(
            image,
            reference_artist=reference_artist,
            brushstroke_report=brushstroke_report,
            style_report=style_report,
        )

    return AnalysisResult(
        image_path=image_path,
        style=style_report,
        brushstrokes=brushstroke_report,
        attribution=attribution_report,
        forgery=forgery_report,
    )
