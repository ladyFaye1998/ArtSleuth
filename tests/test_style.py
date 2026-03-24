"""Tests for the style classification module."""

from __future__ import annotations

from artsleuth.core.style import (
    PERIODS,
    SCHOOLS,
    TECHNIQUES,
    StylePrediction,
    StyleReport,
)

import numpy as np


class TestTaxonomies:
    """Verify style taxonomy completeness and alignment with weights."""

    def test_periods_match_wikiart_styles(self) -> None:
        assert len(PERIODS) == 27

    def test_techniques_match_wikiart_genres(self) -> None:
        assert len(TECHNIQUES) == 11

    def test_schools_nonempty(self) -> None:
        assert len(SCHOOLS) >= 5

    def test_no_duplicates(self) -> None:
        for taxonomy in [PERIODS, SCHOOLS, TECHNIQUES]:
            assert len(taxonomy) == len(set(taxonomy))


class TestStylePrediction:
    """Verify StylePrediction data structure."""

    def test_creation(self) -> None:
        pred = StylePrediction(
            label="Baroque",
            confidence=0.85,
            top_k=[("Baroque", 0.85), ("Mannerism Late Renaissance", 0.10)],
        )
        assert pred.label == "Baroque"
        assert pred.confidence == 0.85
        assert len(pred.top_k) == 2


class TestStyleReport:
    """Verify StyleReport construction."""

    def test_full_report(self) -> None:
        report = StyleReport(
            period=StylePrediction(label="Baroque", confidence=0.9, top_k=[]),
            school=StylePrediction(label="Venetian", confidence=0.7, top_k=[]),
            technique=StylePrediction(label="Landscape", confidence=0.95, top_k=[]),
            embedding=np.zeros(768),
        )
        assert report.period.label == "Baroque"
        assert report.embedding.shape == (768,)
