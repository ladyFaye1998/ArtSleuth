"""Tests for the attribution module."""

from __future__ import annotations

import numpy as np

from artsleuth.core.attribution import (
    AttributionAnalyzer,
    AttributionReport,
    CandidateAttribution,
)
from artsleuth.config import AnalysisConfig


class TestCandidateAttribution:
    """Verify CandidateAttribution data structure."""

    def test_creation(self) -> None:
        ca = CandidateAttribution(
            artist="Artemisia Gentileschi",
            score=0.82,
            confidence_interval=(0.75, 0.89),
            supporting_features=["Strong overall feature correspondence."],
        )
        assert ca.artist == "Artemisia Gentileschi"
        assert ca.score == 0.82
        assert ca.confidence_interval[0] < ca.confidence_interval[1]


class TestAttributionReport:
    """Verify report defaults and accessors."""

    def test_empty_report(self) -> None:
        report = AttributionReport()
        assert report.consensus_artist == "Unknown"
        assert report.consensus_confidence == 0.0
        assert report.multi_hand_flag is False


class TestScoreCandidates:
    """Verify the static scoring method."""

    def test_empty_reference(self) -> None:
        query = np.random.randn(512).astype(np.float32)
        candidates = AttributionAnalyzer._score_candidates(query, {}, top_k=3)
        assert len(candidates) == 1
        assert "no reference gallery" in candidates[0].artist.lower()

    def test_single_reference(self) -> None:
        query = np.random.randn(128).astype(np.float32)
        query /= np.linalg.norm(query)
        reference = {"Test Artist": query.copy()}

        candidates = AttributionAnalyzer._score_candidates(query, reference, top_k=1)
        assert len(candidates) == 1
        assert candidates[0].artist == "Test Artist"
        assert candidates[0].score > 0.5

    def test_multiple_references(self) -> None:
        dim = 128
        rng = np.random.RandomState(42)
        query = rng.randn(dim).astype(np.float32)
        reference = {
            f"Artist_{i}": rng.randn(dim).astype(np.float32) for i in range(5)
        }
        # Make one reference very similar
        reference["Similar Artist"] = query + rng.randn(dim).astype(np.float32) * 0.01

        candidates = AttributionAnalyzer._score_candidates(query, reference, top_k=3)
        assert len(candidates) == 3
        assert candidates[0].artist == "Similar Artist"
