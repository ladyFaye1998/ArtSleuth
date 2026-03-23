"""Tests for the forgery detection module."""

from __future__ import annotations

import numpy as np

from artsleuth.config import AnalysisConfig
from artsleuth.core.forgery import ForgeryDetector, ForgeryReport


class TestForgeryReport:
    """Verify report structure and defaults."""

    def test_default_report(self) -> None:
        report = ForgeryReport()
        assert report.anomaly_score == 0.0
        assert report.is_flagged is False
        assert report.reference_artist == "Unknown"
        assert report.screening_status == "not_configured"

    def test_flagged_report(self) -> None:
        report = ForgeryReport(anomaly_score=0.95, is_flagged=True)
        assert report.is_flagged


class TestForgeryDetector:
    """Verify forgery detection logic."""

    def test_no_reference_returns_unflagged(self) -> None:
        config = AnalysisConfig(device="cpu")
        detector = ForgeryDetector(config)

        # No reference corpus → should return gracefully
        from PIL import Image

        img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
        report = detector.detect(img, reference_artist="Vermeer")

        assert not report.is_flagged
        assert report.screening_status == "not_configured"
        assert len(report.indicators) > 0

    def test_fit_and_detect_inlier(self) -> None:
        config = AnalysisConfig(device="cpu")
        detector = ForgeryDetector(config)

        rng = np.random.RandomState(42)
        corpus = rng.randn(20, 10).astype(np.float64)
        detector.fit_reference("Test Artist", corpus)

        # Query that is close to the corpus mean → should NOT flag
        from unittest.mock import patch as mock_patch

        query_features = corpus.mean(axis=0) + rng.randn(10) * 0.01

        with mock_patch.object(
            ForgeryDetector,
            "_extract_features",
            return_value=query_features,
        ):
            from PIL import Image

            img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
            report = detector.detect(img, reference_artist="Test Artist")

        assert report.anomaly_score < 0.5
        assert report.screening_status == "completed"

    def test_fit_and_detect_outlier(self) -> None:
        config = AnalysisConfig(device="cpu")
        detector = ForgeryDetector(config)

        rng = np.random.RandomState(42)
        corpus = rng.randn(20, 10).astype(np.float64)
        detector.fit_reference("Test Artist", corpus)

        outlier = corpus.mean(axis=0) + 50.0

        from unittest.mock import patch as mock_patch

        with mock_patch.object(
            ForgeryDetector,
            "_extract_features",
            return_value=outlier,
        ):
            from PIL import Image

            img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
            report = detector.detect(img, reference_artist="Test Artist")

        assert report.anomaly_score > 0.5
        assert report.screening_status == "completed"
