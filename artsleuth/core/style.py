"""
Style classification — period, school, and genre.

Three axes: *period* (Renaissance, Baroque, …), *school* (Venetian,
Flemish, …), and *genre* (portrait, landscape, …).

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
        Subject genre prediction (portrait, landscape, etc.).
    embedding:
        The raw CLIP embedding vector, useful for downstream similarity
        searches across a corpus.
    """

    period: StylePrediction
    school: StylePrediction
    technique: StylePrediction
    embedding: np.ndarray


# --- Classifier -------------------------------------------------------------


_TEXT_EMB_CACHE: dict[str, torch.Tensor] = {}
_TOKENIZER_CACHE: list = []

# Prompt templates tuned for CLIP's pre-training distribution
_PERIOD_PROMPT = "a painting in the {} style"
_SCHOOL_PROMPT = "a painting from the {} school of art"
_GENRE_PROMPT = "a {} painting"
_ARTIST_PROMPT = "a painting by {}"

# Well-known artists for zero-shot estimation
KNOWN_ARTISTS: list[str] = [
    "Leonardo da Vinci", "Raphael", "Michelangelo", "Titian",
    "Sandro Botticelli", "Giovanni Bellini", "Jan van Eyck",
    "Albrecht Dürer", "Hieronymus Bosch", "Hans Holbein",
    "El Greco", "Tintoretto", "Caravaggio", "Rembrandt",
    "Vermeer", "Peter Paul Rubens", "Diego Velázquez",
    "Artemisia Gentileschi", "Frans Hals",
    "Jean-Antoine Watteau", "François Boucher",
    "Francisco Goya", "Eugène Delacroix", "J.M.W. Turner",
    "Caspar David Friedrich", "John Constable",
    "Gustave Courbet", "Jean-François Millet", "Édouard Manet",
    "Claude Monet", "Pierre-Auguste Renoir", "Edgar Degas",
    "Camille Pissarro", "Alfred Sisley",
    "Vincent van Gogh", "Paul Cézanne", "Paul Gauguin",
    "Henri de Toulouse-Lautrec", "Georges Seurat",
    "Gustav Klimt", "Alphonse Mucha",
    "Henri Matisse", "André Derain",
    "Pablo Picasso", "Georges Braque",
    "Edvard Munch", "Ernst Ludwig Kirchner", "Egon Schiele",
    "Wassily Kandinsky", "Piet Mondrian", "Paul Klee",
    "Jackson Pollock", "Mark Rothko", "Willem de Kooning",
    "Andy Warhol", "Roy Lichtenstein",
    "Salvador Dalí", "René Magritte", "Max Ernst",
    "Frida Kahlo", "Henri Rousseau", "Amedeo Modigliani",
    "Katsushika Hokusai", "Utagawa Hiroshige",
    "Edward Hopper", "Norman Rockwell", "Georgia O'Keeffe",
    "Marc Chagall", "Joan Miró", "Francis Bacon",
]


class StyleClassifier:
    """Classifies artworks by period, school, and genre using CLIP
    zero-shot classification.

    Compares the CLIP image embedding against text embeddings of
    category descriptions (e.g. "a painting in the Baroque style").
    No trained classification heads required — this leverages CLIP's
    pre-trained vision-language alignment directly.

    Parameters
    ----------
    config:
        Analysis configuration.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config
        self._device = config.resolve_device()
        self._backbone: torch.nn.Module | None = None

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
            Period, school, and genre predictions with confidence
            scores, plus the raw CLIP embedding.
        """
        embedding = self._encode(image)

        period = self._zero_shot_axis(embedding, PERIODS, _PERIOD_PROMPT, top_k)
        school = self._zero_shot_axis(embedding, SCHOOLS, _SCHOOL_PROMPT, top_k)
        technique = self._zero_shot_axis(embedding, TECHNIQUES, _GENRE_PROMPT, top_k)

        return StyleReport(
            period=period,
            school=school,
            technique=technique,
            embedding=embedding.cpu().numpy(),
        )

    def estimate_artist(
        self, embedding_np: np.ndarray, top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Zero-shot artist estimation from a CLIP embedding.

        Returns a ranked list of ``(artist, probability)`` tuples.
        """
        emb = torch.from_numpy(embedding_np).float().to(self._device)
        text_embs = self._get_text_embeddings(KNOWN_ARTISTS, _ARTIST_PROMPT)

        emb_norm = emb / (emb.norm() + 1e-12)
        similarity = emb_norm @ text_embs.T
        probs = F.softmax(similarity / 0.02, dim=-1)

        values, indices = torch.topk(probs, min(top_k, len(KNOWN_ARTISTS)))
        return [
            (KNOWN_ARTISTS[int(idx)], float(val))
            for val, idx in zip(values, indices)
        ]

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

    def _get_text_embeddings(
        self, labels: list[str], prompt_template: str,
    ) -> torch.Tensor:
        """Encode text labels via CLIP text encoder, with caching."""
        cache_key = prompt_template + "\x00" + "\x00".join(labels)
        if cache_key in _TEXT_EMB_CACHE:
            return _TEXT_EMB_CACHE[cache_key].to(self._device)

        backbone = self._ensure_backbone()
        if not hasattr(backbone, "encode_text"):
            raise RuntimeError(
                "CLIP text encoder not available; zero-shot requires "
                "the HuggingFace CLIPModel backend."
            )

        tokenizer = self._get_tokenizer()
        texts = [prompt_template.format(label) for label in labels]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            text_embs = backbone.encode_text(**inputs)

        _TEXT_EMB_CACHE[cache_key] = text_embs.cpu()
        return text_embs

    def _get_tokenizer(self):
        """Lazy-load and cache the CLIP tokenizer."""
        if _TOKENIZER_CACHE:
            return _TOKENIZER_CACHE[0]

        from transformers import AutoTokenizer

        hf_name = "openai/clip-vit-large-patch14"
        tokenizer = AutoTokenizer.from_pretrained(
            hf_name,
            cache_dir=str(self._config.cache_dir) if self._config.cache_dir else None,
        )
        _TOKENIZER_CACHE.append(tokenizer)
        return tokenizer

    def _zero_shot_axis(
        self,
        embedding: torch.Tensor,
        labels: list[str],
        prompt_template: str,
        top_k: int,
    ) -> StylePrediction:
        """Classify one axis via CLIP text-image cosine similarity."""
        text_embs = self._get_text_embeddings(labels, prompt_template)

        emb_norm = embedding / (embedding.norm() + 1e-12)
        similarity = emb_norm @ text_embs.T
        probs = F.softmax(similarity / 0.02, dim=-1)

        top_values, top_indices = torch.topk(probs, min(top_k, len(labels)))
        top_k_list = [
            (labels[int(idx)], float(val))
            for val, idx in zip(top_values, top_indices)
        ]

        best_label, best_conf = top_k_list[0]
        return StylePrediction(
            label=best_label, confidence=best_conf, top_k=top_k_list,
        )
