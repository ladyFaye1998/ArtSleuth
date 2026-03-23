"""
Anomaly-based forgery detection.

A forger can copy what they *see*, but they can't copy what they don't
know they're missing.  An artist's motor habits, material quirks, the
way they instinctively load a brush — these produce statistical
regularities across thousands of feature dimensions that no forger
fully reproduces.  Van Meegeren fooled the experts, but a good embedding
space would likely have flagged the inconsistencies.

One-class anomaly detection against a reference corpus.  Deliberately
conservative — a high score means "this warrants closer examination by
a conservator," not "this is fake."  Attribution history has enough
confident calls that turned out wrong; cautious language matters.

References
----------
van Dantzig, M. M. (1973). *Pictology: An Analytical Method for
    Attribution and Evaluation of Pictures*.
Schölkopf, B. et al. (2001). Estimating the Support of a High-
    Dimensional Distribution. *Neural Computation*, 13(7), 1443–1471.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

    from artsleuth.config import AnalysisConfig
    from artsleuth.core.brushstroke import BrushstrokeReport
    from artsleuth.core.style import StyleReport


# --- Data Structures --------------------------------------------------------


@dataclass(frozen=True)
class AnomalyIndicator:
    """A single dimension along which the query departs from the reference.

    Attributes
    ----------
    feature_name:
        Human-readable name of the feature axis.
    z_score:
        Number of standard deviations from the reference mean.
    description:
        Narrative explanation suitable for an art-historical audience.
    """

    feature_name: str
    z_score: float
    description: str


@dataclass
class ForgeryReport:
    """Forgery screening results for a query painting.

    Attributes
    ----------
    anomaly_score:
        Composite anomaly score (0–1).  Values above 0.7 warrant
        further technical examination; values above 0.9 indicate
        strong statistical departure from the reference corpus.
    is_flagged:
        ``True`` when the anomaly score exceeds the configured
        confidence threshold.
    indicators:
        Ranked list of the feature dimensions contributing most
        to the anomaly score.
    reference_artist:
        The artist against whose corpus the query was screened.
    """

    anomaly_score: float = 0.0
    is_flagged: bool = False
    indicators: list[AnomalyIndicator] = field(default_factory=list)
    reference_artist: str = "Unknown"


# --- Detector ---------------------------------------------------------------


class ForgeryDetector:
    """Screens artworks for statistical anomalies relative to a reference corpus.

    The detector implements a two-stage pipeline:

    1. **Feature aggregation** — Brushstroke descriptors and style
       embeddings are concatenated into a holistic feature vector.
    2. **Anomaly scoring** — The feature vector is evaluated against
       a one-class model (Mahalanobis distance in the learned feature
       space) fitted to the reference corpus.

    Parameters
    ----------
    config:
        Analysis configuration.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config
        self._reference_stats: dict[str, _ReferenceStats] | None = None

    # --- Public API ---------------------------------------------------------

    def detect(
        self,
        image: "Image.Image",
        reference_artist: str,
        brushstroke_report: BrushstrokeReport | None = None,
        style_report: StyleReport | None = None,
    ) -> ForgeryReport:
        """Screen a painting against a specific artist's reference corpus.

        Parameters
        ----------
        image:
            RGB artwork image.
        reference_artist:
            Name of the artist whose corpus defines "normal."
        brushstroke_report:
            Pre-computed brushstroke analysis.
        style_report:
            Pre-computed style classification.

        Returns
        -------
        ForgeryReport
            Anomaly score, flag status, and contributing indicators.
        """
        features = self._extract_features(brushstroke_report, style_report)
        stats = self._ensure_reference_stats(reference_artist)

        if stats is None:
            return ForgeryReport(
                anomaly_score=0.0,
                is_flagged=False,
                indicators=[
                    AnomalyIndicator(
                        feature_name="reference_corpus",
                        z_score=0.0,
                        description=(
                            f"No reference corpus available for '{reference_artist}'. "
                            "Load reference data to enable forgery screening."
                        ),
                    )
                ],
                reference_artist=reference_artist,
            )

        anomaly_score, indicators = self._compute_anomaly(features, stats)

        return ForgeryReport(
            anomaly_score=anomaly_score,
            is_flagged=anomaly_score > self._config.confidence_threshold,
            indicators=indicators,
            reference_artist=reference_artist,
        )

    def fit_reference(self, artist: str, feature_vectors: np.ndarray) -> None:
        """Fit a one-class reference model from a corpus of known works.

        Parameters
        ----------
        artist:
            Artist name for this reference corpus.
        feature_vectors:
            Array of shape ``(n_works, feature_dim)`` containing
            aggregated feature vectors for authenticated works.
        """
        if self._reference_stats is None:
            self._reference_stats = {}

        mean = feature_vectors.mean(axis=0)
        cov = np.cov(feature_vectors, rowvar=False)

        # Regularise — small corpora produce singular covariance matrices,
        # and the last thing we want is a LinAlgError mid-attribution
        cov += np.eye(cov.shape[0]) * 1e-6

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        self._reference_stats[artist] = _ReferenceStats(
            mean=mean, cov_inv=cov_inv, std=np.sqrt(np.diag(cov))
        )

    # --- Internal Methods ---------------------------------------------------

    @staticmethod
    def _extract_features(
        brushstroke_report: BrushstrokeReport | None,
        style_report: StyleReport | None,
    ) -> np.ndarray:
        """Fuse available analyses into a single feature vector."""
        parts: list[np.ndarray] = []

        if style_report is not None:
            parts.append(style_report.embedding)

        if brushstroke_report is not None and brushstroke_report.descriptors:
            stroke_features = np.array(
                [
                    [d.orientation, d.coherence, d.energy, d.curvature]
                    for d in brushstroke_report.descriptors
                ]
            )
            parts.append(stroke_features.mean(axis=0))

        if not parts:
            return np.zeros(1, dtype=np.float32)

        return np.concatenate(parts).astype(np.float64)

    def _ensure_reference_stats(self, artist: str) -> "_ReferenceStats | None":
        """Look up pre-fitted reference statistics."""
        if self._reference_stats is None:
            self._reference_stats = {}
        return self._reference_stats.get(artist)

    @staticmethod
    def _compute_anomaly(
        features: np.ndarray,
        stats: "_ReferenceStats",
    ) -> tuple[float, list[AnomalyIndicator]]:
        """Compute anomaly score via Mahalanobis distance."""
        dim = min(features.shape[0], stats.mean.shape[0])
        f = features[:dim]
        m = stats.mean[:dim]
        ci = stats.cov_inv[:dim, :dim]
        std = stats.std[:dim]

        diff = f - m
        mahal_sq = float(diff @ ci @ diff)
        mahal = np.sqrt(max(mahal_sq, 0.0))

        # Squash to 0–1 via sigmoid — human-readable beats mathematically pure
        anomaly_score = float(1.0 / (1.0 + np.exp(-0.1 * (mahal - 10.0))))

        # Per-feature z-scores for interpretability
        z_scores = np.abs(diff) / (std + 1e-12)
        top_indices = np.argsort(z_scores)[::-1][:5]

        indicators = []
        for idx in top_indices:
            name = f"dim_{idx}"
            z = float(z_scores[idx])
            desc = _describe_anomaly(name, z)
            indicators.append(AnomalyIndicator(feature_name=name, z_score=z, description=desc))

        return anomaly_score, indicators


# --- Helpers ----------------------------------------------------------------


@dataclass(frozen=True)
class _ReferenceStats:
    """Pre-computed statistics for a reference artist corpus."""

    mean: np.ndarray
    cov_inv: np.ndarray
    std: np.ndarray


def _describe_anomaly(feature_name: str, z_score: float) -> str:
    """Generate a human-readable anomaly description."""
    severity = (
        "substantially" if z_score > 3.0 else "moderately" if z_score > 2.0 else "slightly"
    )
    return (
        f"Embedding dimension {feature_name} departs {severity} "
        f"(z = {z_score:.1f}) from the reference corpus."
    )
