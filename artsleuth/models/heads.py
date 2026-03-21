"""
Task-specific classification and projection heads.

Each analytical task in ArtSleuth is served by a lightweight linear
head that projects the backbone's embedding space into the target
label space.  Heads are designed to be independently trainable: a
researcher can fine-tune the style head on a new corpus without
affecting the brushstroke analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from artsleuth.models.backbones import embedding_dim

logger = logging.getLogger(__name__)


# --- Public API -------------------------------------------------------------


def build_style_heads(
    *,
    period_classes: int,
    school_classes: int,
    technique_classes: int,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> dict[str, nn.Linear]:
    """Construct (or load) the three style-classification heads.

    If pre-trained weights are available on HuggingFace, they are loaded
    automatically.  Otherwise, heads are initialised with Kaiming uniform
    weights suitable for immediate fine-tuning.

    Parameters
    ----------
    period_classes:
        Number of period categories.
    school_classes:
        Number of school categories.
    technique_classes:
        Number of technique categories.
    device:
        PyTorch device string.
    cache_dir:
        Directory for cached weights.

    Returns
    -------
    dict[str, nn.Linear]
        Mapping from axis name to its classification head.
    """
    from artsleuth.config import BackboneType

    dim = embedding_dim(BackboneType.CLIP)

    heads = {
        "period": nn.Linear(dim, period_classes),
        "school": nn.Linear(dim, school_classes),
        "technique": nn.Linear(dim, technique_classes),
    }

    # Attempt to load pre-trained weights
    loaded = _try_load_pretrained_heads(heads, cache_dir)
    if not loaded:
        logger.info(
            "No pre-trained style heads found; initialised with random weights. "
            "Fine-tune on a labelled corpus for meaningful predictions."
        )
        for head in heads.values():
            nn.init.kaiming_uniform_(head.weight, nonlinearity="linear")
            nn.init.zeros_(head.bias)

    for head in heads.values():
        head.eval()
        head.to(device)

    return heads


def build_attribution_head(
    *,
    embedding_dim_combined: int,
    device: str = "cpu",
) -> nn.Linear:
    """Build a projection head for attribution embedding comparison.

    Parameters
    ----------
    embedding_dim_combined:
        Dimensionality of the fused feature vector (style + brushstroke).
    device:
        PyTorch device string.

    Returns
    -------
    nn.Linear
        Projection head that maps the fused vector to a unit-normalised
        comparison space.
    """
    head = nn.Linear(embedding_dim_combined, 256)
    nn.init.kaiming_uniform_(head.weight, nonlinearity="linear")
    nn.init.zeros_(head.bias)
    head.eval()
    return head.to(device)


# --- Internal ---------------------------------------------------------------


def _try_load_pretrained_heads(
    heads: dict[str, nn.Linear],
    cache_dir: Path | None,
) -> bool:
    """Attempt to load pre-trained head weights from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download

        for axis_name, head in heads.items():
            weight_path = hf_hub_download(
                repo_id="ladyFaye1998/artsleuth-weights",
                filename=f"style_{axis_name}.pt",
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            head.load_state_dict(state)
            logger.info("Loaded pre-trained %s head.", axis_name)

        return True
    except Exception:
        return False
