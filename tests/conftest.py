"""Shared test fixtures for ArtSleuth."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from artsleuth.config import AnalysisConfig


@pytest.fixture()
def sample_image() -> Image.Image:
    """A synthetic 256×256 RGB image simulating a painting surface."""
    rng = np.random.RandomState(42)
    arr = rng.randint(80, 220, (256, 256, 3), dtype=np.uint8)
    # Add simulated brushstrokes as directional gradients
    for y in range(0, 256, 16):
        arr[y : y + 2, :, :] = rng.randint(40, 100, (2, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture()
def small_image() -> Image.Image:
    """A minimal 64×64 RGB test image."""
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture()
def config() -> AnalysisConfig:
    """Default analysis configuration for tests."""
    return AnalysisConfig(
        device="cpu",
        patch_size=64,
        max_resolution=256,
    )
