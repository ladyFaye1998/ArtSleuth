"""Tests for the brushstroke analysis module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from artsleuth.core.brushstroke import BrushstrokeAnalyzer, BrushstrokeReport, StrokeDescriptor


class TestStrokeDescriptor:
    """Verify StrokeDescriptor immutability and field access."""

    def test_creation(self) -> None:
        desc = StrokeDescriptor(
            orientation=0.5,
            coherence=0.8,
            energy=1.2,
            curvature=0.3,
            embedding=np.zeros(384),
            bbox=(0, 0, 64, 64),
        )
        assert desc.orientation == 0.5
        assert desc.coherence == 0.8
        assert desc.bbox == (0, 0, 64, 64)

    def test_frozen(self) -> None:
        desc = StrokeDescriptor(
            orientation=0.0, coherence=0.0, energy=0.0, curvature=0.0,
            embedding=np.zeros(1), bbox=(0, 0, 1, 1),
        )
        with pytest.raises(AttributeError):
            desc.orientation = 1.0  # type: ignore[misc]


class TestStructureTensor:
    """Verify structure-tensor computation on synthetic patches."""

    def test_horizontal_gradient(self) -> None:
        patch = torch.zeros(3, 64, 64)
        for col in range(64):
            patch[:, :, col] = col / 64.0

        _orientation, coherence, energy, _curvature = (
            BrushstrokeAnalyzer._structure_tensor_stats(patch)
        )
        assert energy > 0, "Gradient should produce nonzero energy."
        assert 0 <= coherence <= 1, "Coherence must be in [0, 1]."

    def test_uniform_patch(self) -> None:
        patch = torch.full((3, 64, 64), 0.5)
        _, _coherence, energy, _ = BrushstrokeAnalyzer._structure_tensor_stats(patch)
        assert energy < 0.01, "Uniform patch should have near-zero energy."

    def test_diagonal_gradient(self) -> None:
        patch = torch.zeros(3, 64, 64)
        for i in range(64):
            for j in range(64):
                patch[:, i, j] = (i + j) / 128.0

        orientation, _coherence, energy, _ = (
            BrushstrokeAnalyzer._structure_tensor_stats(patch)
        )
        assert energy > 0
        assert abs(orientation - np.pi / 4) < 0.5 or abs(orientation + np.pi / 4) < 0.5


class TestBrushstrokeReport:
    """Verify report aggregation defaults."""

    def test_empty_report(self) -> None:
        report = BrushstrokeReport()
        assert report.mean_coherence == 0.0
        assert len(report.descriptors) == 0
        assert report.orientation_histogram.shape == (36,)
