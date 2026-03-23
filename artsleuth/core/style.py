"""
Style classification — period, school, and technique.

Three axes: *period* (Renaissance, Baroque, …), *school* (Venetian,
Flemish, …), and *technique* (oil on canvas, tempera on panel, …).

These labels are culturally constructed — "Baroque" isn't a pixel value,
it's a consensus built over 400 years of art-historical writing.  CLIP
works well here because its vision-language pre-training already encodes
this kind of socially grounded concept, unlike purely visual backbones.

References
----------
Radford, A. et al. (2021). Learning Transferable Visual Models from
    Natural Language Supervision. *ICML*.
Wölfflin, H. (1915). *Principles of Art History*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from PIL import Image

    from artsleuth.config import AnalysisConfig


# --- Taxonomies -------------------------------------------------------------

# WikiArt style taxonomy (27 classes) — aligned with pretrained weights
PERIODS: list[str] = [
    "Abstract Expressionism",
    "Action Painting",
    "Analytical Cubism",
    "Art Nouveau",
    "Baroque",
    "Color Field Painting",
    "Contemporary Realism",
    "Cubism",
    "Early Renaissance",
    "Expressionism",
    "Fauvism",
    "High Renaissance",
    "Impressionism",
    "Mannerism Late Renaissance",
    "Minimalism",
    "Naive Art Primitivism",
    "New Realism",
    "Northern Renaissance",
    "Pointillism",
    "Pop Art",
    "Post Impressionism",
    "Realism",
    "Rococo",
    "Romanticism",
    "Symbolism",
    "Synthetic Cubism",
    "Ukiyo e",
]

# No pretrained weights — these predictions use randomly initialised heads until fine-tuned
SCHOOLS: list[str] = [
    "Florentine",
    "Venetian",
    "Roman",
    "Flemish",
    "Dutch Golden Age",
    "Spanish",
    "French Academic",
    "Pre-Raphaelite",
    "Barbizon",
    "Hudson River",
    "Bauhaus",
    "De Stijl",
    "Vienna Secession",
]

# WikiArt genre taxonomy (11 classes) — aligned with pretrained weights
TECHNIQUES: list[str] = [
    "Abstract Painting",
    "Cityscape",
    "Genre Painting",
    "Illustration",
    "Landscape",
    "Nude Painting",
    "Portrait",
    "Religious Painting",
    "Sketch And Study",
    "Still Life",
    "Unknown Genre",
]


# --- Data Structures --------------------------------------------------------


@dataclass(frozen=True)
class StylePrediction:
    """A single style-axis prediction with associated confidence.

    Attributes
    ----------
    label:
        Predicted category name.
    confidence:
        Softmax probability for this label.
    top_k:
        Ranked list of ``(label, confidence)`` for the top-k alternatives.
    """

    label: str
    confidence: float
    top_k: list[tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class StyleReport:
    """Complete style classification for an artwork.

    Attributes
    ----------
    period:
        Chronological period prediction.
    school:
        Art-historical school prediction.
    technique:
        Material technique prediction.
    embedding:
        The raw CLIP embedding vector, useful for downstream similarity
        searches across a corpus.
    """

    period: StylePrediction
    school: StylePrediction
    technique: StylePrediction
    embedding: np.ndarray


# --- Classifier -------------------------------------------------------------


class StyleClassifier:
    """Classifies artworks by period, school, and technique.

    The classifier uses CLIP as its feature backbone and applies three
    independent linear projection heads — one per style axis.  These
    heads can be loaded from pre-trained weights on HuggingFace or
    fine-tuned on a user-provided corpus.

    Parameters
    ----------
    config:
        Analysis configuration.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config
        self._device = config.resolve_device()
        self._backbone: torch.nn.Module | None = None
        self._heads: dict[str, torch.nn.Linear] | None = None

    # --- Public API ---------------------------------------------------------

    def classify(self, image: Image.Image, top_k: int = 5) -> StyleReport:
        """Classify an artwork across all three style axes.

        Parameters
        ----------
        image:
            RGB artwork image.
        top_k:
            Number of alternative predictions to include per axis.

        Returns
        -------
        StyleReport
            Period, school, and technique predictions with confidence
            scores, plus the raw CLIP embedding.
        """
        embedding = self._encode(image)
        heads = self._ensure_heads()

        period = self._predict_axis(embedding, heads["period"], PERIODS, top_k)
        school = self._predict_axis(embedding, heads["school"], SCHOOLS, top_k)
        technique = self._predict_axis(embedding, heads["technique"], TECHNIQUES, top_k)

        return StyleReport(
            period=period,
            school=school,
            technique=technique,
            embedding=embedding.cpu().numpy(),
        )

    # --- Internal Methods ---------------------------------------------------

    def _ensure_backbone(self) -> torch.nn.Module:
        """Lazy-load the CLIP vision encoder."""
        if self._backbone is None:
            from artsleuth.models.backbones import load_backbone
            from artsleuth.config import BackboneType

            self._backbone = load_backbone(
                BackboneType.CLIP,
                device=self._device,
                cache_dir=self._config.cache_dir,
            )
        return self._backbone

    def _ensure_heads(self) -> dict[str, torch.nn.Linear]:
        """Lazy-load or initialise the classification heads."""
        if self._heads is None:
            from artsleuth.models.heads import build_style_heads

            self._heads = build_style_heads(
                period_classes=len(PERIODS),
                school_classes=len(SCHOOLS),
                technique_classes=len(TECHNIQUES),
                device=self._device,
                cache_dir=self._config.cache_dir,
            )
        return self._heads

    def _encode(self, image: Image.Image) -> torch.Tensor:
        """Extract a CLIP embedding from the input image."""
        from artsleuth.preprocessing.transforms import prepare_for_backbone
        from artsleuth.config import BackboneType

        tensor = prepare_for_backbone(
            image,
            BackboneType.CLIP,
            self._config.max_resolution,
            enable_art_preprocessing=self._config.enable_art_preprocessing,
        )
        tensor = tensor.unsqueeze(0).to(self._device)

        backbone = self._ensure_backbone()
        with torch.no_grad():
            embedding = backbone(tensor)

        return embedding.squeeze(0)

    @staticmethod
    def _predict_axis(
        embedding: torch.Tensor,
        head: torch.nn.Linear,
        labels: list[str],
        top_k: int,
    ) -> StylePrediction:
        """Run a single classification head and return a StylePrediction."""
        with torch.no_grad():
            logits = head(embedding)
            probs = F.softmax(logits, dim=-1)

        top_values, top_indices = torch.topk(probs, min(top_k, len(labels)))
        top_k_list = [
            (labels[idx], float(val)) for val, idx in zip(top_values, top_indices)
        ]

        best_label, best_conf = top_k_list[0]
        return StylePrediction(label=best_label, confidence=best_conf, top_k=top_k_list)
