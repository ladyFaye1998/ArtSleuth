"""
Task-specific classification and projection heads.

Thin linear layers on top of the backbone — one per task, deliberately
kept independent so fine-tuning the style head on a new corpus won't
affect the brushstroke analysis.  The backbone does the heavy lifting;
these project its embeddings into task-specific label spaces.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from artsleuth.models.backbones import embedding_dim

logger = logging.getLogger(__name__)

_PACKAGE_WEIGHTS_DIR = Path(__file__).parent / "weights"

# Local checkpoint filenames (under artsleuth/models/weights/).
_LOCAL_HEAD_FILES: dict[str, str] = {
    "period": "style_head.pt",
    "technique": "genre_head.pt",
}


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

    Pre-trained weights are loaded from ``artsleuth/models/weights/`` when
    present, otherwise from Hugging Face Hub where available.  Any head still
    without weights is initialised with Kaiming uniform values suitable for
    fine-tuning.

    Parameters
    ----------
    period_classes:
        Number of period categories.
    school_classes:
        Number of school categories.
    technique_classes:
        Number of genre categories (called ``technique`` internally for
        backward compatibility with saved checkpoint keys).
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

    loaded_axes = _try_load_pretrained_heads(heads, cache_dir, backbone_embedding_dim=dim)

    if not loaded_axes:
        logger.info(
            "No pre-trained style heads found; initialised with random weights. "
            "Fine-tune on a labelled corpus for meaningful predictions."
        )
        for head in heads.values():
            nn.init.kaiming_uniform_(head.weight, nonlinearity="linear")
            nn.init.zeros_(head.bias)
    else:
        for axis_name, head in heads.items():
            if axis_name not in loaded_axes:
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


def _linear_shapes_from_state(state: dict) -> tuple[int, int] | None:
    """Return (in_features, out_features) for a Linear state dict, or None."""
    weight = state.get("weight")
    if weight is None or not hasattr(weight, "shape"):
        return None
    if weight.ndim != 2:
        return None
    out_f, in_f = int(weight.shape[0]), int(weight.shape[1])
    return in_f, out_f


def _load_state_dict_from_file(path: Path) -> dict[str, torch.Tensor] | None:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning("Failed to load weights from %s: %s", path, exc)
        return None


def _apply_pretrained_state(
    heads: dict[str, nn.Linear],
    axis_name: str,
    state: dict,
    backbone_embedding_dim: int,
) -> bool:
    """Load checkpoint into head, rebuilding the layer if shapes differ."""
    shapes = _linear_shapes_from_state(state)
    if shapes is None:
        logger.warning(
            "Pretrained %s head state dict has no valid weight matrix; skipping.",
            axis_name,
        )
        return False

    sd_in, sd_out = shapes
    head = heads[axis_name]
    expected_in = head.in_features
    expected_out = head.out_features

    if sd_in == expected_in and sd_out == expected_out:
        try:
            head.load_state_dict(state)
        except Exception as exc:
            logger.warning(
                "Failed to load state dict for %s head (shape matched): %s",
                axis_name,
                exc,
            )
            return False
    else:
        logger.warning(
            "Pretrained %s head weight shape (%d → %d) does not match configured "
            "head (%d → %d); rebuilding layer to match checkpoint.",
            axis_name,
            sd_in,
            sd_out,
            expected_in,
            expected_out,
        )
        replacement = nn.Linear(sd_in, sd_out)
        try:
            replacement.load_state_dict(state)
        except Exception as exc:
            logger.warning(
                "Failed to load state dict for rebuilt %s head: %s",
                axis_name,
                exc,
            )
            return False
        heads[axis_name] = replacement

    if sd_in != backbone_embedding_dim:
        logger.warning(
            "Pretrained %s head expects input dimension %d, but current CLIP "
            "backbone embedding dimension is %d.",
            axis_name,
            sd_in,
            backbone_embedding_dim,
        )
    if sd_out != expected_out:
        logger.warning(
            "Pretrained %s head has %d output classes; model config expects %d.",
            axis_name,
            sd_out,
            expected_out,
        )

    return True


def _try_load_axis_from_hub(
    heads: dict[str, nn.Linear],
    axis_name: str,
    cache_dir: Path | None,
    backbone_embedding_dim: int,
) -> bool:
    try:
        from huggingface_hub import hf_hub_download

        weight_path = hf_hub_download(
            repo_id="ladyFaye1998/artsleuth-weights",
            filename=f"style_{axis_name}.pt",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            logger.warning("Hugging Face artifact for %s head is not a state dict.", axis_name)
            return False
        ok = _apply_pretrained_state(heads, axis_name, state, backbone_embedding_dim)
        if ok:
            logger.info("Loaded pretrained %s head from Hugging Face Hub.", axis_name)
        return ok
    except Exception as exc:
        logger.debug("Hugging Face load failed for %s head: %s", axis_name, exc)
        return False


def _try_load_pretrained_heads(
    heads: dict[str, nn.Linear],
    cache_dir: Path | None,
    *,
    backbone_embedding_dim: int,
) -> set[str]:
    """Load pre-trained heads from local package weights, then Hugging Face.

    Returns the set of axis names for which weights were loaded successfully.
    ``school`` is never loaded from a checkpoint; callers should initialise it
    separately when it is absent from this set.
    """
    logger.info("No pretrained weights for 'school' head; using random initialisation.")

    loaded: set[str] = set()

    for axis_name in ("period", "technique"):
        head = heads[axis_name]
        local_name = _LOCAL_HEAD_FILES.get(axis_name)
        local_path = _PACKAGE_WEIGHTS_DIR / local_name if local_name else None

        if local_path is not None and local_path.is_file():
            state = _load_state_dict_from_file(local_path)
            if (
                state is not None
                and isinstance(state, dict)
                and _apply_pretrained_state(heads, axis_name, state, backbone_embedding_dim)
            ):
                loaded.add(axis_name)
                logger.info(
                    "Loaded pretrained %s head from local weights.",
                    axis_name,
                )
                continue

        if _try_load_axis_from_hub(
            heads, axis_name, cache_dir, backbone_embedding_dim
        ):
            loaded.add(axis_name)

    return loaded
