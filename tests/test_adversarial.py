"""Tests for artsleuth.core.adversarial."""

from __future__ import annotations

import numpy as np
from PIL import Image

from artsleuth.core.adversarial import ForgerySimulator, ForgeryTechnique


def _random_image(size: tuple[int, int] = (100, 100)) -> Image.Image:
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_forgery_technique_creation() -> None:
    ft = ForgeryTechnique(
        name="test_tech", description="A test technique.", severity=0.5,
    )
    assert ft.name == "test_tech"
    assert ft.severity == 0.5


def test_artificial_aging() -> None:
    sim = ForgerySimulator()
    img = _random_image()
    result = sim.artificial_aging(img)

    assert isinstance(result, Image.Image)
    assert result.size == img.size
    assert np.array(result).tolist() != np.array(img).tolist()


def test_style_transfer_perturbation() -> None:
    sim = ForgerySimulator()
    img = _random_image()
    result = sim.style_transfer_perturbation(img)

    assert isinstance(result, Image.Image)
    assert result.size == img.size
    assert np.array(result).tolist() != np.array(img).tolist()


def test_material_anachronism() -> None:
    sim = ForgerySimulator()
    img = _random_image()
    result = sim.material_anachronism(img)

    assert isinstance(result, Image.Image)
    assert result.size == img.size
    assert np.array(result).tolist() != np.array(img).tolist()


def test_composite_forgery() -> None:
    sim = ForgerySimulator()
    img = _random_image()
    result = sim.composite_forgery(img)

    assert isinstance(result, Image.Image)
    assert result.size == img.size
    assert np.array(result).tolist() != np.array(img).tolist()


def test_available_techniques() -> None:
    sim = ForgerySimulator()
    techniques = sim.available_techniques()

    assert isinstance(techniques, list)
    assert len(techniques) >= 4
    assert all(isinstance(t, ForgeryTechnique) for t in techniques)


def test_severity_range() -> None:
    sim = ForgerySimulator()
    img = _random_image()

    result_low = sim.artificial_aging(img, severity=0.0)
    result_high = sim.artificial_aging(img, severity=1.0)

    assert isinstance(result_low, Image.Image)
    assert isinstance(result_high, Image.Image)
    assert result_low.size == img.size
    assert result_high.size == img.size
