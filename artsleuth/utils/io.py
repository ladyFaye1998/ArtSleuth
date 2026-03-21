"""
Image I/O and format handling.

Loads artwork images from files, URLs, or byte streams.  Handles the
annoying stuff automatically — EXIF rotation, alpha stripping, greyscale
promotion — so you never have to debug a sideways Vermeer again.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Formats that ArtSleuth can ingest.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_image(source: str | Path | BinaryIO) -> Image.Image:
    """Load an artwork image and normalise it to RGB.

    Handles EXIF rotation, alpha-channel removal, and greyscale
    promotion transparently.

    Parameters
    ----------
    source:
        A file path, URL string, or file-like byte stream.

    Returns
    -------
    Image.Image
        RGB PIL image.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file format is unsupported.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = Image.open(path)
    else:
        image = Image.open(source)

    image = ImageOps.exif_transpose(image) or image
    return image.convert("RGB")


def load_image_from_url(url: str, timeout: int = 30) -> Image.Image:
    """Download and load an image from a URL.

    Parameters
    ----------
    url:
        HTTP(S) URL pointing to an image file.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    Image.Image
        RGB PIL image.
    """
    import urllib.request

    logger.info("Downloading image from %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()

    return load_image(BytesIO(data))


def save_image(
    image: Image.Image,
    path: str | Path,
    *,
    quality: int = 95,
) -> Path:
    """Save an image to disk with sensible defaults.

    Parameters
    ----------
    image:
        PIL image to save.
    path:
        Output path; format is inferred from extension.
    quality:
        JPEG quality (ignored for lossless formats).

    Returns
    -------
    Path
        The resolved output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(out), quality=quality)
    logger.info("Saved image to %s", out)
    return out
