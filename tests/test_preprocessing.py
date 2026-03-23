"""Tests for the preprocessing module."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from artsleuth.config import BackboneType
from artsleuth.preprocessing.patches import extract_patches
from artsleuth.preprocessing.transforms import (
    correct_varnish,
    normalise_canvas_texture,
    prepare_for_backbone,
    suppress_craquelure,
)


class TestPrepareForBackbone:
    """Verify backbone-specific preprocessing."""

    def test_dinov2_output_shape(self) -> None:
        img = Image.fromarray(np.full((300, 400, 3), 128, dtype=np.uint8))
        tensor = prepare_for_backbone(img, BackboneType.DINO_V2, max_resolution=1024)
        assert tensor.shape == (3, 518, 518)

    def test_clip_output_shape(self) -> None:
        img = Image.fromarray(np.full((300, 400, 3), 128, dtype=np.uint8))
        tensor = prepare_for_backbone(img, BackboneType.CLIP, max_resolution=1024)
        assert tensor.shape == (3, 224, 224)

    def test_normalisation_range(self) -> None:
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:50, :, :] = 255
        img = Image.fromarray(arr)
        tensor = prepare_for_backbone(img, BackboneType.CLIP, max_resolution=1024)
        assert tensor.min() < 0, "ImageNet normalisation should produce negative values."
        assert tensor.max() < 5, "Values should not be extreme."


class TestCorrectiveTransforms:
    """Verify corrective preprocessing transforms."""

    def test_varnish_correction_reduces_red(self) -> None:
        arr = np.full((50, 50, 3), 200, dtype=np.uint8)
        img = Image.fromarray(arr)
        corrected = correct_varnish(img, strength=1.0)
        corr_arr = np.array(corrected)
        assert corr_arr[0, 0, 0] < 200, "Red channel should decrease."

    def test_craquelure_suppression(self) -> None:
        arr = np.full((50, 50, 3), 128, dtype=np.uint8)
        arr[25, :, :] = 0  # Simulate a crack
        img = Image.fromarray(arr)
        filtered = suppress_craquelure(img, kernel_size=3)
        result = np.array(filtered)
        assert result[25, 25, 0] > 0, "Crack line should be attenuated."

    def test_canvas_texture_normalisation(self) -> None:
        img = Image.fromarray(np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8))
        result = normalise_canvas_texture(img, frequency_cutoff=0.2)
        assert result.size == img.size


class TestPatchExtraction:
    """Verify patch extraction strategies."""

    def test_grid_patches(self) -> None:
        img = Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))
        patches, bboxes = extract_patches(
            img, patch_size=64, strategy="grid", max_resolution=256, overlap=0.0
        )
        assert len(patches) == 16  # 4×4 grid
        assert all(p.shape == (3, 64, 64) for p in patches)
        assert len(bboxes) == len(patches)

    def test_salient_patches(self) -> None:
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        patches, bboxes = extract_patches(
            img, patch_size=64, strategy="salient", max_resolution=256
        )
        assert len(patches) > 0
        assert len(patches) <= 64

    def test_adaptive_patches(self) -> None:
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        patches_adaptive, _ = extract_patches(
            img, patch_size=64, strategy="adaptive", max_resolution=256
        )
        patches_grid, _ = extract_patches(
            img, patch_size=64, strategy="grid", max_resolution=256
        )
        assert len(patches_adaptive) >= len(patches_grid)
