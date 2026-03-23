"""Tests for artsleuth.core.workshop."""

from __future__ import annotations

import numpy as np

from artsleuth.core.workshop import (
    HandAssignment,
    WorkshopDecomposition,
)


def _grid_bboxes(
    rows: int, cols: int, patch_w: int = 50, patch_h: int = 50,
) -> list[tuple[int, int, int, int]]:
    """Generate a regular grid of (x, y, w, h) bounding boxes."""
    bboxes: list[tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            bboxes.append((c * patch_w, r * patch_h, patch_w, patch_h))
    return bboxes


def test_hand_assignment_creation() -> None:
    ha = HandAssignment(
        hand_id=0,
        label="primary_hand",
        confidence=0.95,
        patch_count=10,
        spatial_extent=0.6,
        mean_coherence=0.8,
        mean_energy=0.5,
    )
    assert ha.hand_id == 0
    assert ha.label == "primary_hand"


def test_workshop_single_hand() -> None:
    rng = np.random.RandomState(42)
    center = rng.randn(64) * 0.01
    embeddings = np.tile(center, (20, 1)) + rng.normal(0, 0.001, (20, 64))
    bboxes = _grid_bboxes(4, 5)
    image_size = (250, 200)

    decomp = WorkshopDecomposition(max_hands=4)
    report = decomp.decompose(embeddings, bboxes, image_size)

    assert report.num_hands >= 1
    primary = [a for a in report.assignments if a.label == "primary_hand"]
    assert len(primary) == 1
    total_patches = sum(a.patch_count for a in report.assignments)
    assert total_patches == 20


def test_workshop_multiple_hands() -> None:
    rng = np.random.RandomState(42)
    cluster_a = rng.normal(loc=0.0, scale=0.1, size=(10, 64))
    cluster_b = rng.normal(loc=5.0, scale=0.1, size=(10, 64))
    embeddings = np.vstack([cluster_a, cluster_b])
    bboxes = _grid_bboxes(4, 5)
    image_size = (250, 200)

    decomp = WorkshopDecomposition(max_hands=4)
    report = decomp.decompose(embeddings, bboxes, image_size)

    assert report.num_hands >= 2
    assert report.is_workshop is True


def test_workshop_report_fields() -> None:
    rng = np.random.RandomState(42)
    embeddings = rng.normal(loc=0.0, scale=0.1, size=(20, 64))
    bboxes = _grid_bboxes(4, 5)
    image_size = (250, 200)

    decomp = WorkshopDecomposition(max_hands=4)
    report = decomp.decompose(embeddings, bboxes, image_size)

    assert report.num_hands > 0
    assert len(report.patch_labels) == 20
    assert len(report.assignments) > 0
    primary = [a for a in report.assignments if a.label == "primary_hand"]
    assert len(primary) == 1


def test_hand_map_shape() -> None:
    rng = np.random.RandomState(42)
    embeddings = rng.normal(loc=0.0, scale=0.1, size=(20, 64))
    bboxes = _grid_bboxes(4, 5)
    image_size = (200, 300)

    decomp = WorkshopDecomposition(max_hands=4)
    report = decomp.decompose(embeddings, bboxes, image_size)

    assert report.hand_map is not None
    assert report.hand_map.shape == (300, 200)
