"""Tests for artsleuth.core.temporal."""

from __future__ import annotations

import numpy as np
import pytest

from artsleuth.core.temporal import (
    TemporalReference,
    TemporalRegistry,
    TemporalStyleModel,
)


def test_temporal_reference_creation() -> None:
    emb = np.random.randn(32).astype(np.float32)
    ref = TemporalReference(year=1620, embedding=emb, title="Test")
    assert ref.year == 1620
    assert ref.title == "Test"
    assert ref.embedding.shape == (32,)


def test_temporal_model_requires_min_refs() -> None:
    model = TemporalStyleModel()
    for i in range(2):
        model.add_reference(
            1600 + i * 10,
            np.random.randn(32).astype(np.float32),
        )
    with pytest.raises(ValueError, match="At least"):
        model.fit()


def test_temporal_model_fit_and_predict() -> None:
    rng = np.random.RandomState(0)
    model = TemporalStyleModel()

    for i in range(10):
        year = 1600 + i * 10
        emb = rng.randn(32).astype(np.float64)
        emb[0] = year / 1000.0
        model.add_reference(year, emb)

    model.fit()

    query = rng.randn(32).astype(np.float64)
    query[0] = 1640.0 / 1000.0
    pred = model.predict(query)

    assert 1550 <= pred.estimated_year <= 1730
    assert pred.temporal_score > 0
    assert isinstance(pred.confidence_band, tuple)
    assert len(pred.confidence_band) == 2


def test_temporal_registry() -> None:
    rng = np.random.RandomState(1)
    registry = TemporalRegistry()

    for i in range(5):
        emb = rng.randn(32).astype(np.float64)
        emb[0] = (1610 + i * 10) / 1000.0
        registry.register(
            "Artemisia",
            year=1610 + i * 10,
            embedding=emb,
            title=f"Work {i}",
        )

    registry.fit_all()
    assert "Artemisia" in registry.artists

    unknown_pred = registry.predict(
        "UnknownArtist", rng.randn(32).astype(np.float64)
    )
    assert unknown_pred is None


def test_drift_rate() -> None:
    model = TemporalStyleModel()
    base = np.zeros(32, dtype=np.float64)

    for i, year in enumerate([1600, 1620, 1640]):
        emb = base.copy()
        emb[0] = i * 2.0
        model.add_reference(year, emb)

    model.fit()
    assert model.drift_rate() > 0
