"""
Temporal style drift modelling.

Most attribution systems treat an artist's style as a single static
distribution.  In practice, Artemisia Gentileschi c.1612 painted very
differently from Artemisia c.1652, and confusing the two periods can
distort attribution scores.

This module fits a Gaussian process over time-stamped reference
embeddings for each artist, modelling the expected trajectory of their
style through embedding space.  Given a query painting, it estimates
the most likely date of execution and reports temporal plausibility as a
separate signal (it does not modify attribution scores directly).

References
----------
Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes
    for Machine Learning*. MIT Press.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import MinMaxScaler

if TYPE_CHECKING:
    pass


# --- Data Structures --------------------------------------------------------


@dataclass(frozen=True)
class TemporalReference:
    """A single dated reference work.

    Attributes
    ----------
    year:
        Year of execution.
    embedding:
        Feature vector (e.g. 512-dim from CLIP or the fusion module).
    title:
        Optional work title for provenance tracking.
    """

    year: int
    embedding: np.ndarray
    title: str = ""


@dataclass(frozen=True)
class TemporalPrediction:
    """Result of temporal analysis for a query embedding.

    Attributes
    ----------
    estimated_year:
        Maximum-likelihood date of execution estimated by the GP.
    confidence_band:
        95 % credible interval ``(year_low, year_high)`` around the
        estimated year.
    temporal_score:
        Plausibility score in [0, 1] indicating how well the query
        fits the artist's trajectory at the estimated date.
    drift_rate:
        Magnitude of style change per decade in embedding space,
        averaged across the reference timeline.
    """

    estimated_year: float
    confidence_band: tuple[float, float]
    temporal_score: float
    drift_rate: float


# --- Single-Artist Model ---------------------------------------------------


_MIN_REFERENCES = 3


class TemporalStyleModel:
    """Models a single artist's style evolution over time.

    A Gaussian process is fit from year to PCA-reduced embedding
    space.  At inference the model searches over the artist's active
    period to find the year whose predicted embedding best matches
    the query, yielding an estimated date and a temporal plausibility
    score.

    The model requires at least three dated reference works before
    fitting.
    """

    def __init__(self) -> None:
        self._references: list[TemporalReference] = []
        self._gp_fitted: bool = False
        self._gp: GaussianProcessRegressor | None = None
        self._pca: PCA | None = None
        self._scaler: MinMaxScaler | None = None
        self._year_min: float = 0.0
        self._year_max: float = 0.0

    # --- Public API ---------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the GP has been successfully fit."""
        return self._gp_fitted

    def add_reference(
        self,
        year: int,
        embedding: np.ndarray,
        title: str = "",
    ) -> None:
        """Register a dated reference work.

        Parameters
        ----------
        year:
            Year of execution.
        embedding:
            Feature vector for the work.
        title:
            Optional human-readable title.
        """
        self._references.append(
            TemporalReference(year=year, embedding=embedding, title=title)
        )
        self._gp_fitted = False

    def fit(self) -> None:
        """Fit the Gaussian process over the current references.

        Raises
        ------
        ValueError
            If fewer than ``_MIN_REFERENCES`` references have been
            registered.
        """
        if len(self._references) < _MIN_REFERENCES:
            raise ValueError(
                f"At least {_MIN_REFERENCES} references are required to fit "
                f"the temporal model (got {len(self._references)})."
            )

        years = np.array([r.year for r in self._references], dtype=np.float64)
        embeddings = np.stack([r.embedding for r in self._references])

        self._year_min = float(years.min())
        self._year_max = float(years.max())

        # Normalize years to [0, 1] for numerical stability
        self._scaler = MinMaxScaler()
        years_norm = self._scaler.fit_transform(years.reshape(-1, 1))

        # PCA reduction — keep min(20, n_samples, dim) components
        n_components = min(20, len(self._references), embeddings.shape[1])
        self._pca = PCA(n_components=n_components)
        embeddings_pca = self._pca.fit_transform(embeddings)

        # GP: RBF + WhiteKernel for observation noise
        kernel = RBF(length_scale=0.3) + WhiteKernel(noise_level=0.1)
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
        )
        self._gp.fit(years_norm, embeddings_pca)
        self._gp_fitted = True

    def predict(self, embedding: np.ndarray) -> TemporalPrediction:
        """Estimate execution date and temporal plausibility of a query.

        Parameters
        ----------
        embedding:
            Feature vector for the query painting.

        Returns
        -------
        TemporalPrediction
            Estimated year, confidence band, plausibility score, and
            drift rate.

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        """
        self._check_fitted()
        assert self._gp is not None
        assert self._pca is not None
        assert self._scaler is not None

        query_pca = self._pca.transform(embedding.reshape(1, -1))

        # Grid search over the artist's active period (+-10 years)
        lo = self._year_min - 10.0
        hi = self._year_max + 10.0
        candidate_years = np.arange(lo, hi + 1, 1.0)
        candidate_norm = self._scaler.transform(
            candidate_years.reshape(-1, 1)
        )

        predicted, predicted_std = self._gp.predict(
            candidate_norm, return_std=True,
        )

        # Distance between each predicted embedding and the query
        diffs = predicted - query_pca  # (n_candidates, n_pca)
        distances = np.linalg.norm(diffs, axis=1)

        best_idx = int(np.argmin(distances))
        estimated_year = float(candidate_years[best_idx])

        # 95 % credible interval from GP predictive std
        mean_std = float(np.mean(predicted_std))
        half_width = 1.96 * mean_std * (self._year_max - self._year_min)
        half_width = max(half_width, 1.0)
        confidence_band = (
            estimated_year - half_width,
            estimated_year + half_width,
        )

        # Temporal plausibility score
        median_ref_dist = self._median_reference_distance()
        scale = 2.0 * median_ref_dist ** 2 if median_ref_dist > 0 else 1.0
        temporal_score = float(
            math.exp(-(distances[best_idx] ** 2) / scale)
        )

        return TemporalPrediction(
            estimated_year=estimated_year,
            confidence_band=confidence_band,
            temporal_score=temporal_score,
            drift_rate=self.drift_rate(),
        )

    def drift_rate(self) -> float:
        """Mean embedding-space distance per decade across the timeline.

        Returns
        -------
        float
            Average L2 drift per ten-year interval.  Zero if fewer than
            two references exist.
        """
        if len(self._references) < 2:
            return 0.0

        sorted_refs = sorted(self._references, key=lambda r: r.year)
        total_distance = 0.0
        total_years = 0.0

        for prev, curr in zip(sorted_refs[:-1], sorted_refs[1:]):
            gap = float(curr.year - prev.year)
            if gap <= 0:
                continue
            dist = float(
                np.linalg.norm(curr.embedding - prev.embedding)
            )
            total_distance += dist
            total_years += gap

        if total_years == 0:
            return 0.0
        return (total_distance / total_years) * 10.0

    # --- Internal Methods ---------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if the model has not been fit."""
        if not self._gp_fitted:
            raise RuntimeError(
                "TemporalStyleModel has not been fit.  "
                "Call .fit() after adding references."
            )

    def _median_reference_distance(self) -> float:
        """Median pairwise L2 distance among PCA-projected references."""
        assert self._pca is not None
        embeddings = np.stack([r.embedding for r in self._references])
        projected = self._pca.transform(embeddings)

        n = projected.shape[0]
        if n < 2:
            return 1.0

        distances: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                distances.append(
                    float(np.linalg.norm(projected[i] - projected[j]))
                )
        return float(np.median(distances))


# --- Multi-Artist Registry -------------------------------------------------


class TemporalRegistry:
    """Manages :class:`TemporalStyleModel` instances for multiple artists.

    Provides a thin convenience layer so that calling code does not
    need to track per-artist model objects directly.
    """

    def __init__(self) -> None:
        self._models: dict[str, TemporalStyleModel] = {}

    # --- Public API ---------------------------------------------------------

    @property
    def artists(self) -> list[str]:
        """Names of artists whose models have been successfully fit."""
        return [
            name
            for name, model in self._models.items()
            if model.is_fitted
        ]

    def register(
        self,
        artist: str,
        year: int,
        embedding: np.ndarray,
        title: str = "",
    ) -> None:
        """Add a dated reference work for *artist*.

        Parameters
        ----------
        artist:
            Artist or workshop name.
        year:
            Year of execution.
        embedding:
            Feature vector for the work.
        title:
            Optional human-readable title.
        """
        if artist not in self._models:
            self._models[artist] = TemporalStyleModel()
        self._models[artist].add_reference(year, embedding, title)

    def fit_all(self) -> None:
        """Fit temporal models for all artists with enough references.

        Artists with fewer than ``_MIN_REFERENCES`` dated works are
        silently skipped.
        """
        for model in self._models.values():
            if len(model._references) >= _MIN_REFERENCES:
                model.fit()

    def predict(
        self,
        artist: str,
        embedding: np.ndarray,
    ) -> TemporalPrediction | None:
        """Estimate execution date for a query attributed to *artist*.

        Parameters
        ----------
        artist:
            Artist name (must match a previously registered name).
        embedding:
            Feature vector for the query painting.

        Returns
        -------
        TemporalPrediction or None
            ``None`` if the artist has no fitted model.
        """
        model = self._models.get(artist)
        if model is None or not model.is_fitted:
            return None
        return model.predict(embedding)


# --- Heuristic Date Estimation from Style Classification -------------------

_PERIOD_DATE_RANGES: dict[str, tuple[int, int]] = {
    "Abstract Expressionism": (1940, 1965),
    "Action Painting": (1945, 1960),
    "Analytical Cubism": (1909, 1912),
    "Art Nouveau": (1890, 1910),
    "Baroque": (1590, 1750),
    "Color Field Painting": (1950, 1975),
    "Contemporary Realism": (1970, 2020),
    "Cubism": (1907, 1925),
    "Early Renaissance": (1400, 1500),
    "Expressionism": (1905, 1935),
    "Fauvism": (1900, 1910),
    "High Renaissance": (1490, 1530),
    "Impressionism": (1860, 1890),
    "Mannerism Late Renaissance": (1520, 1600),
    "Minimalism": (1960, 1975),
    "Naive Art Primitivism": (1880, 1940),
    "New Realism": (1960, 1970),
    "Northern Renaissance": (1430, 1570),
    "Pointillism": (1884, 1910),
    "Pop Art": (1955, 1975),
    "Post Impressionism": (1880, 1910),
    "Realism": (1840, 1900),
    "Rococo": (1720, 1780),
    "Romanticism": (1780, 1850),
    "Symbolism": (1880, 1910),
    "Synthetic Cubism": (1912, 1925),
    "Ukiyo e": (1670, 1900),
}


def estimate_date_from_style(
    period_top_k: list[tuple[str, float]],
) -> TemporalPrediction:
    """Estimate a date range from period classification softmax outputs.

    Uses a weighted mixture of the date ranges for the top predicted
    periods, producing a probability-weighted midpoint and confidence
    band.  This is a heuristic — no GP or reference data required.
    """
    weighted_lo = 0.0
    weighted_hi = 0.0
    total_weight = 0.0

    for label, conf in period_top_k:
        date_range = _PERIOD_DATE_RANGES.get(label)
        if date_range is None:
            continue
        lo, hi = date_range
        weighted_lo += lo * conf
        weighted_hi += hi * conf
        total_weight += conf

    if total_weight == 0:
        return TemporalPrediction(
            estimated_year=1700.0,
            confidence_band=(1400.0, 2000.0),
            temporal_score=0.1,
            drift_rate=0.0,
        )

    est_lo = weighted_lo / total_weight
    est_hi = weighted_hi / total_weight
    midpoint = (est_lo + est_hi) / 2.0
    half_span = (est_hi - est_lo) / 2.0

    top_conf = period_top_k[0][1] if period_top_k else 0.0
    plausibility = min(top_conf * 1.2, 1.0)

    return TemporalPrediction(
        estimated_year=midpoint,
        confidence_band=(est_lo - half_span * 0.3, est_hi + half_span * 0.3),
        temporal_score=plausibility,
        drift_rate=0.0,
    )
