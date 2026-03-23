"""
Intelligent patch extraction for brushstroke analysis.

Dividing a painting into local patches is the foundational step of
brushstroke analysis — the question is *which* patches matter.  A naïve
grid wastes time on flat sky passages that carry little information.
A purely saliency-based approach might miss subtle background work
where the workshop hand is most visible.

Three strategies, pick your trade-off:

  * **Grid** — uniform tiles, fast, deterministic, slightly boring.
  * **Salient** — chases the most textured regions.  Great for
    expressive passages, blind to quiet ones.
  * **Adaptive** (default) — grid plus extra sampling of high-detail
    areas.  Best of both worlds, usually.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torchvision import transforms

if TYPE_CHECKING:
    from PIL import Image

    from artsleuth.config import PatchStrategy


# --- Constants --------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# --- Public API -------------------------------------------------------------


def extract_patches(
    image: Image.Image,
    *,
    patch_size: int = 224,
    strategy: PatchStrategy = "adaptive",
    max_resolution: int = 2048,
    overlap: float = 0.25,
) -> tuple[list[torch.Tensor], list[tuple[int, int, int, int]]]:
    """Extract analysis patches from an artwork image.

    Parameters
    ----------
    image:
        RGB PIL image.
    patch_size:
        Side length (in pixels) of each square patch.
    strategy:
        Extraction strategy name.
    max_resolution:
        Maximum side length; the image is downscaled if necessary.
    overlap:
        Fractional overlap between adjacent grid patches (0–0.5).

    Returns
    -------
    patches:
        List of normalised tensors, each of shape ``(3, patch_size, patch_size)``.
    bboxes:
        Corresponding bounding boxes as ``(x, y, width, height)`` in the
        (possibly downscaled) image coordinate frame.
    """
    from artsleuth.preprocessing.transforms import _clamp_resolution

    image = _clamp_resolution(image, max_resolution).convert("RGB")
    strategy_str = strategy if isinstance(strategy, str) else strategy.value

    if strategy_str == "grid":
        return _grid_patches(image, patch_size, overlap)
    elif strategy_str == "salient":
        return _salient_patches(image, patch_size)
    else:
        return _adaptive_patches(image, patch_size, overlap)


# --- Strategies -------------------------------------------------------------


def _grid_patches(
    image: Image.Image,
    patch_size: int,
    overlap: float,
) -> tuple[list[torch.Tensor], list[tuple[int, int, int, int]]]:
    """Uniform grid tiling with configurable overlap."""
    w, h = image.size
    step = max(1, int(patch_size * (1 - overlap)))

    patches: list[torch.Tensor] = []
    bboxes: list[tuple[int, int, int, int]] = []

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )

    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            crop = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(to_tensor(crop))
            bboxes.append((x, y, patch_size, patch_size))

    return patches, bboxes


def _salient_patches(
    image: Image.Image,
    patch_size: int,
    max_patches: int = 64,
) -> tuple[list[torch.Tensor], list[tuple[int, int, int, int]]]:
    """Select patches from regions of highest gradient energy."""
    arr = np.array(image.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(arr)
    energy = np.sqrt(gx**2 + gy**2)

    w, h = image.size
    candidates: list[tuple[float, int, int]] = []

    step = patch_size // 2
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            patch_energy = energy[y : y + patch_size, x : x + patch_size].mean()
            candidates.append((patch_energy, x, y))

    candidates.sort(key=lambda c: c[0], reverse=True)
    selected = candidates[:max_patches]

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )

    patches: list[torch.Tensor] = []
    bboxes: list[tuple[int, int, int, int]] = []

    for _, x, y in selected:
        crop = image.crop((x, y, x + patch_size, y + patch_size))
        patches.append(to_tensor(crop))
        bboxes.append((x, y, patch_size, patch_size))

    return patches, bboxes


def _adaptive_patches(
    image: Image.Image,
    patch_size: int,
    overlap: float,
    salient_boost: int = 16,
) -> tuple[list[torch.Tensor], list[tuple[int, int, int, int]]]:
    """Grid tiling augmented with extra salient-region patches."""
    grid_patches, grid_bboxes = _grid_patches(image, patch_size, overlap)
    salient_patches_list, salient_bboxes = _salient_patches(
        image, patch_size, max_patches=salient_boost
    )

    # De-duplicate patches that overlap significantly with grid patches
    existing = set(grid_bboxes)
    for patch, bbox in zip(salient_patches_list, salient_bboxes, strict=False):
        if bbox not in existing:
            grid_patches.append(patch)
            grid_bboxes.append(bbox)
            existing.add(bbox)

    return grid_patches, grid_bboxes
