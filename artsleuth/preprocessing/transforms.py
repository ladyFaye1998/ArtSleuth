"""
Art-specific image transforms.

ImageNet preprocessing assumes clean, uniformly-lit photos.  Paintings
present different challenges: centuries of varnish yellowing, craquelure
crosshatching every surface, canvas weave humming at its own spatial
frequency, and variable gallery lighting or glass reflections.

These transforms try to peel back the noise of age and photography so
the backbone can focus on what the artist actually put there.  Not a
replacement for proper conservation imaging, but a practical first step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torchvision import transforms

if TYPE_CHECKING:
    from PIL import Image

    from artsleuth.config import BackboneType


# --- Backbone-Specific Normalisation ----------------------------------------

# ImageNet statistics used by DINOv2 and CLIP
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def prepare_for_backbone(
    image: "Image.Image",
    backbone_type: "BackboneType",
    max_resolution: int = 2048,
) -> torch.Tensor:
    """Prepare an artwork image for backbone feature extraction.

    Applies resolution clamping, optional varnish correction, and
    backbone-appropriate normalisation.

    Parameters
    ----------
    image:
        RGB PIL image.
    backbone_type:
        Target backbone (affects input resolution and normalisation).
    max_resolution:
        Maximum side length before downscaling.

    Returns
    -------
    torch.Tensor
        Preprocessed tensor of shape ``(3, H, W)``.
    """
    from artsleuth.config import BackboneType

    image = _clamp_resolution(image, max_resolution)

    if backbone_type == BackboneType.DINO_V2:
        target_size = 518  # DINOv2 native resolution
    else:
        target_size = 224  # CLIP standard

    transform = transforms.Compose(
        [
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )

    return transform(image)


# --- Corrective Transforms -------------------------------------------------


def correct_varnish(
    image: "Image.Image",
    strength: float = 0.3,
) -> "Image.Image":
    """Attenuate warm-shifted varnish yellowing.

    Applies a channel-wise correction that reduces the red-yellow bias
    introduced by aged varnish layers, approximating the painting's
    original colour temperature.

    Parameters
    ----------
    image:
        RGB input image.
    strength:
        Correction intensity (0 = no change, 1 = aggressive).

    Returns
    -------
    Image.Image
        Colour-corrected image.
    """
    from PIL import Image as PILImage

    arr = np.array(image, dtype=np.float32)

    # Pull back the warm amber shift — most old varnish pushes R and G up
    arr[..., 0] *= 1.0 - 0.15 * strength  # red
    arr[..., 1] *= 1.0 - 0.05 * strength  # green
    arr[..., 2] *= 1.0 + 0.10 * strength  # blue

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return PILImage.fromarray(arr)


def suppress_craquelure(
    image: "Image.Image",
    kernel_size: int = 3,
) -> "Image.Image":
    """Reduce craquelure (crack) noise via selective median filtering.

    A median filter suppresses the thin, high-contrast crack lines
    without blurring broader brushstroke edges — unlike Gaussian
    smoothing, which indiscriminately attenuates both.

    Parameters
    ----------
    image:
        RGB input image.
    kernel_size:
        Median filter kernel size (must be odd).

    Returns
    -------
    Image.Image
        Filtered image with reduced crack visibility.
    """
    from PIL import ImageFilter, Image as PILImage

    return image.filter(ImageFilter.MedianFilter(size=kernel_size))


def normalise_canvas_texture(
    image: "Image.Image",
    frequency_cutoff: float = 0.1,
) -> "Image.Image":
    """Attenuate periodic canvas-weave texture via frequency-domain filtering.

    The canvas weave produces a regular grid pattern at a frequency
    determined by thread count.  We suppress this band in the Fourier
    domain while preserving the lower-frequency brushstroke information.

    Parameters
    ----------
    image:
        RGB input image.
    frequency_cutoff:
        Normalised cutoff frequency (0–1) below which spatial frequencies
        are preserved.

    Returns
    -------
    Image.Image
        Image with attenuated canvas texture.
    """
    from PIL import Image as PILImage

    arr = np.array(image, dtype=np.float32)
    result = np.zeros_like(arr)

    for c in range(3):
        channel = arr[..., c]
        f_transform = np.fft.fft2(channel)
        f_shifted = np.fft.fftshift(f_transform)

        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        r = int(min(rows, cols) * frequency_cutoff)

        # Create a soft low-pass mask
        y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
        mask = np.exp(-(x * x + y * y) / (2 * (r ** 2 + 1e-6)))

        # Blend: preserve low frequencies, attenuate high
        f_filtered = f_shifted * (0.3 + 0.7 * mask)
        result[..., c] = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))

    result = np.clip(result, 0, 255).astype(np.uint8)
    return PILImage.fromarray(result)


# --- Helpers ----------------------------------------------------------------


def _clamp_resolution(image: "Image.Image", max_side: int) -> "Image.Image":
    """Downscale an image if either side exceeds the maximum."""
    w, h = image.size
    if max(w, h) <= max_side:
        return image

    scale = max_side / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, resample=3)  # BICUBIC
