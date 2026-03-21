"""
Model and reference-gallery registry.

Manages the download, caching, and versioning of pre-trained model
weights and reference embedding galleries from HuggingFace Hub.
The registry provides a single point of access for all remote assets,
ensuring reproducibility through explicit version pinning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# HuggingFace repository containing ArtSleuth model artefacts.
_HF_REPO_ID = "ladyFaye1998/artsleuth-weights"


# --- Public API -------------------------------------------------------------


def load_reference_gallery(
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Load the bundled reference-gallery embeddings.

    The gallery contains pre-computed feature vectors for a curated
    set of well-attributed artworks, enabling out-of-the-box artist
    attribution.  Users can extend this gallery with
    ``AttributionAnalyzer.register_reference()``.

    Parameters
    ----------
    cache_dir:
        Local directory for cached downloads.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from artist name to their representative embedding.
    """
    try:
        from huggingface_hub import hf_hub_download

        gallery_path = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename="reference_gallery.npz",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        data = np.load(gallery_path)
        return {key: data[key] for key in data.files}

    except Exception as exc:
        logger.info(
            "Reference gallery not available (%s). "
            "Attribution will return placeholder results until a "
            "reference corpus is registered.",
            exc,
        )
        return {}


def list_available_models() -> list[dict[str, Any]]:
    """List model artefacts available on HuggingFace Hub.

    Returns
    -------
    list[dict[str, Any]]
        Metadata for each available artefact.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        files = api.list_repo_files(repo_id=_HF_REPO_ID)
        return [{"filename": f, "repo": _HF_REPO_ID} for f in files]
    except Exception:
        logger.warning("Unable to query HuggingFace Hub for available models.")
        return []


def download_model(
    filename: str,
    cache_dir: Path | None = None,
) -> Path:
    """Download a specific model artefact from HuggingFace Hub.

    Parameters
    ----------
    filename:
        Name of the artefact file (e.g. ``"style_period.pt"``).
    cache_dir:
        Local cache directory.

    Returns
    -------
    Path
        Local path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=filename,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
