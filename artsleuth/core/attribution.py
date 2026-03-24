"""
Artist and workshop attribution scoring.

Determining *who* painted a given work is among the most consequential
— and contentious — judgements in art history, with real implications
for market value and scholarly understanding.  Renaissance workshops
complicate things further: a master might design the composition, paint
the key figures, and delegate the background to an assistant.

Traditional connoisseurship boils down to visual memory: "this hand
recalls Artemisia's action nodes in the Uffizi *Judith*."  Beautiful
intuition, terrible reproducibility.  The goal here is to keep the
intuition and add the reproducibility — a learned metric space where
paintings by the same hand cluster together and paintings by different
hands separate.  Confidence intervals included, because sometimes
even the embedding is uncertain, and that's worth knowing.

References
----------
Ainsworth, M. W. (2005). From Connoisseurship to Technical Art History.
    *Getty Research Journal*, 159–176.
Lesin, D. (2025). The Gentileschi Debate Through Machine Eyes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL import Image

    from artsleuth.config import AnalysisConfig
    from artsleuth.core.brushstroke import BrushstrokeReport
    from artsleuth.core.style import StyleReport


# --- Data Structures --------------------------------------------------------


@dataclass(frozen=True)
class CandidateAttribution:
    """Attribution score for a single candidate artist.

    Attributes
    ----------
    artist:
        Candidate artist or workshop name.
    score:
        Probability estimate (0–1) that the query painting is by this hand.
    confidence_interval:
        95 % confidence interval (normal approximation) around the score,
        reflecting epistemic uncertainty in limited reference corpora.
    supporting_features:
        Human-readable list of qualitative feature-correspondence descriptions.
    """

    artist: str
    score: float
    confidence_interval: tuple[float, float]
    supporting_features: list[str] = field(default_factory=list)


@dataclass
class AttributionReport:
    """Complete attribution analysis for a query painting.

    Attributes
    ----------
    candidates:
        Ranked list of candidate attributions, highest score first.
    consensus_artist:
        Top-ranked candidate (convenience accessor).
    consensus_confidence:
        Score of the top-ranked candidate.
    multi_hand_flag:
        ``True`` when brushstroke clustering suggests more than one
        distinct hand contributed to the painting — a hallmark of
        workshop production.
    """

    candidates: list[CandidateAttribution] = field(default_factory=list)
    consensus_artist: str = "Unknown"
    consensus_confidence: float = 0.0
    multi_hand_flag: bool = False


# --- Analyzer ---------------------------------------------------------------


class AttributionAnalyzer:
    """Scores candidate artist attributions for a query painting.

    The analyzer operates in an embedding-comparison paradigm:

    1. The query painting is encoded via the same backbone used for
       brushstroke and style analysis.
    2. The resulting embedding is compared against a reference gallery
       of known-attribution works using cosine similarity.
    3. Similarities are calibrated into probability estimates via
       temperature-scaled softmax.
    4. Confidence intervals are derived via normal approximation.

    Parameters
    ----------
    config:
        Analysis configuration.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config
        self._device = config.resolve_device()
        self._reference_embeddings: dict[str, np.ndarray] | None = None

    # --- Public API ---------------------------------------------------------

    def attribute(
        self,
        image: "Image.Image",
        brushstroke_report: BrushstrokeReport | None = None,
        style_report: StyleReport | None = None,
        top_k: int = 5,
    ) -> AttributionReport:
        """Produce an attribution report for a query painting.

        Parameters
        ----------
        image:
            RGB artwork image.
        brushstroke_report:
            Pre-computed brushstroke analysis (avoids redundant computation
            when running the full pipeline).
        style_report:
            Pre-computed style classification.
        top_k:
            Number of candidate artists to return.

        Returns
        -------
        AttributionReport
            Ranked candidates with calibrated scores and confidence intervals.
        """
        query_embedding = self._build_query_embedding(image, brushstroke_report, style_report)
        reference = self._ensure_reference_gallery()
        candidates = self._score_candidates(query_embedding, reference, top_k)

        multi_hand = (
            brushstroke_report is not None
            and brushstroke_report.cluster_labels is not None
            and len(set(brushstroke_report.cluster_labels.tolist())) > 1
        )

        if candidates:
            return AttributionReport(
                candidates=candidates,
                consensus_artist=candidates[0].artist,
                consensus_confidence=candidates[0].score,
                multi_hand_flag=multi_hand,
            )
        return AttributionReport(multi_hand_flag=multi_hand)

    def register_reference(self, artist: str, embedding: np.ndarray) -> None:
        """Add or update a reference embedding for a known artist.

        Parameters
        ----------
        artist:
            Artist or workshop name.
        embedding:
            Pre-computed embedding vector from the ArtSleuth backbone.
        """
        if self._reference_embeddings is None:
            self._reference_embeddings = {}
        self._reference_embeddings[artist] = embedding

    # --- Internal Methods ---------------------------------------------------

    def _ensure_reference_gallery(self) -> dict[str, np.ndarray]:
        """Load the reference gallery, initialising with bundled data if needed."""
        if self._reference_embeddings is None:
            self._reference_embeddings = self._load_bundled_references()
        return self._reference_embeddings

    @staticmethod
    def _load_bundled_references() -> dict[str, np.ndarray]:
        """Attempt to load bundled reference embeddings from HuggingFace.

        Returns an empty dict if no pre-computed gallery is available,
        which causes attribution to return low-confidence results and
        a warning suggesting the user build a custom reference set.
        """
        try:
            from artsleuth.models.registry import load_reference_gallery

            return load_reference_gallery()
        except Exception:
            return {}

    def _build_query_embedding(
        self,
        image: "Image.Image",
        brushstroke_report: BrushstrokeReport | None,
        style_report: StyleReport | None,
    ) -> np.ndarray:
        """Fuse available feature vectors into a unified query embedding."""
        components: list[np.ndarray] = []

        if style_report is not None:
            components.append(style_report.embedding)

        if brushstroke_report is not None and brushstroke_report.descriptors:
            stroke_embs = np.stack([d.embedding for d in brushstroke_report.descriptors])
            components.append(stroke_embs.mean(axis=0))

        if not components:
            from artsleuth.preprocessing.transforms import prepare_for_backbone
            from artsleuth.models.backbones import load_backbone

            tensor = prepare_for_backbone(
                image,
                self._config.backbone,
                self._config.max_resolution,
                enable_art_preprocessing=self._config.enable_art_preprocessing,
            )
            tensor = tensor.unsqueeze(0).to(self._device)
            backbone = load_backbone(
                self._config.backbone,
                device=self._device,
                cache_dir=self._config.cache_dir,
            )
            with torch.no_grad():
                feat = backbone(tensor)
            components.append(feat.squeeze(0).cpu().numpy())

        combined = np.concatenate(components)
        norm = np.linalg.norm(combined) + 1e-12
        return combined / norm

    @staticmethod
    def _score_candidates(
        query: np.ndarray,
        reference: dict[str, np.ndarray],
        top_k: int,
        temperature: float = 0.07,
    ) -> list[CandidateAttribution]:
        """Score candidates via temperature-scaled cosine similarity."""
        if not reference:
            return [
                CandidateAttribution(
                    artist="Unknown (no reference gallery loaded)",
                    score=0.0,
                    confidence_interval=(0.0, 0.0),
                    supporting_features=["Load a reference gallery for meaningful attribution."],
                )
            ]

        artists = list(reference.keys())
        ref_matrix = np.stack([reference[a] for a in artists])

        # Truncate or pad to match query dimensionality
        dim = min(query.shape[0], ref_matrix.shape[1])
        q = query[:dim]
        r = ref_matrix[:, :dim]

        norms_r = np.linalg.norm(r, axis=1, keepdims=True) + 1e-12
        r_normed = r / norms_r

        similarities = r_normed @ q
        scaled = similarities / temperature
        exp_s = np.exp(scaled - scaled.max())
        probs = exp_s / exp_s.sum()

        ranked = np.argsort(probs)[::-1][:top_k]

        candidates = []
        for idx in ranked:
            score = float(probs[idx])
            margin = 1.96 * np.sqrt(score * (1 - score) / max(len(artists), 1))
            candidates.append(
                CandidateAttribution(
                    artist=artists[idx],
                    score=score,
                    confidence_interval=(
                        max(0.0, score - margin),
                        min(1.0, score + margin),
                    ),
                    supporting_features=_identify_supporting_features(similarities[idx]),
                )
            )

        return candidates


def _identify_supporting_features(similarity: float) -> list[str]:
    """Map a similarity score to qualitative supporting evidence."""
    features = []
    if similarity > 0.8:
        features.append("Strong overall feature correspondence.")
    elif similarity > 0.5:
        features.append("Moderate feature correspondence.")
    else:
        features.append("Weak feature correspondence; attribution uncertain.")
    return features


# --- Heuristic Artist Estimation from Style Classification -----------------

_ARTIST_PROFILES: list[tuple[str, list[str], list[str]]] = [
    ("Leonardo da Vinci", ["High Renaissance"], ["Portrait", "Religious Painting"]),
    ("Raphael", ["High Renaissance"], ["Portrait", "Religious Painting"]),
    ("Michelangelo", ["High Renaissance"], ["Religious Painting", "Nude Painting"]),
    ("Titian", ["High Renaissance"], ["Portrait", "Religious Painting"]),
    ("Sandro Botticelli", ["Early Renaissance"], ["Religious Painting"]),
    ("Giovanni Bellini", ["Early Renaissance"], ["Religious Painting"]),
    ("Fra Angelico", ["Early Renaissance"], ["Religious Painting"]),
    ("Jan van Eyck", ["Northern Renaissance"], ["Portrait", "Religious Painting"]),
    ("Albrecht Dürer", ["Northern Renaissance"], ["Portrait", "Religious Painting"]),
    ("Hieronymus Bosch", ["Northern Renaissance"], ["Religious Painting", "Genre Painting"]),
    ("Hans Holbein the Younger", ["Northern Renaissance"], ["Portrait"]),
    ("El Greco", ["Mannerism Late Renaissance"], ["Religious Painting", "Portrait"]),
    ("Tintoretto", ["Mannerism Late Renaissance"], ["Religious Painting"]),
    ("Caravaggio", ["Baroque"], ["Religious Painting", "Genre Painting"]),
    ("Rembrandt", ["Baroque"], ["Portrait", "Religious Painting"]),
    ("Vermeer", ["Baroque"], ["Genre Painting", "Portrait"]),
    ("Peter Paul Rubens", ["Baroque"], ["Religious Painting", "Portrait", "Nude Painting"]),
    ("Diego Velázquez", ["Baroque"], ["Portrait", "Genre Painting"]),
    ("Artemisia Gentileschi", ["Baroque"], ["Religious Painting", "Genre Painting"]),
    ("Frans Hals", ["Baroque"], ["Portrait"]),
    ("Jean-Antoine Watteau", ["Rococo"], ["Genre Painting"]),
    ("François Boucher", ["Rococo"], ["Genre Painting", "Nude Painting"]),
    ("Jean-Honoré Fragonard", ["Rococo"], ["Genre Painting"]),
    ("Francisco Goya", ["Romanticism"], ["Portrait", "Genre Painting"]),
    ("Eugène Delacroix", ["Romanticism"], ["Religious Painting", "Genre Painting"]),
    ("J.M.W. Turner", ["Romanticism"], ["Landscape"]),
    ("Caspar David Friedrich", ["Romanticism"], ["Landscape"]),
    ("Gustave Courbet", ["Realism"], ["Landscape", "Portrait", "Genre Painting"]),
    ("Jean-François Millet", ["Realism"], ["Genre Painting", "Landscape"]),
    ("Édouard Manet", ["Realism", "Impressionism"], ["Portrait", "Genre Painting"]),
    ("Claude Monet", ["Impressionism"], ["Landscape"]),
    ("Pierre-Auguste Renoir", ["Impressionism"], ["Portrait", "Genre Painting", "Nude Painting"]),
    ("Edgar Degas", ["Impressionism"], ["Genre Painting", "Portrait"]),
    ("Camille Pissarro", ["Impressionism"], ["Landscape", "Cityscape"]),
    ("Alfred Sisley", ["Impressionism"], ["Landscape"]),
    ("Vincent van Gogh", ["Post Impressionism"], ["Landscape", "Portrait", "Still Life"]),
    ("Paul Cézanne", ["Post Impressionism"], ["Landscape", "Still Life", "Portrait"]),
    ("Paul Gauguin", ["Post Impressionism"], ["Genre Painting", "Landscape"]),
    ("Henri de Toulouse-Lautrec", ["Post Impressionism"], ["Genre Painting", "Portrait"]),
    ("Georges Seurat", ["Pointillism"], ["Landscape", "Genre Painting"]),
    ("Gustav Klimt", ["Art Nouveau", "Symbolism"], ["Portrait", "Nude Painting"]),
    ("Alphonse Mucha", ["Art Nouveau"], ["Illustration", "Portrait"]),
    ("Henri Matisse", ["Fauvism"], ["Portrait", "Landscape", "Still Life"]),
    ("André Derain", ["Fauvism"], ["Landscape", "Portrait"]),
    ("Pablo Picasso", ["Cubism", "Analytical Cubism", "Synthetic Cubism"], ["Portrait", "Abstract Painting"]),
    ("Georges Braque", ["Cubism", "Analytical Cubism"], ["Still Life", "Abstract Painting"]),
    ("Edvard Munch", ["Expressionism", "Symbolism"], ["Portrait", "Genre Painting"]),
    ("Ernst Ludwig Kirchner", ["Expressionism"], ["Landscape", "Portrait"]),
    ("Wassily Kandinsky", ["Abstract Expressionism", "Expressionism"], ["Abstract Painting"]),
    ("Piet Mondrian", ["Abstract Expressionism"], ["Abstract Painting"]),
    ("Jackson Pollock", ["Action Painting"], ["Abstract Painting"]),
    ("Mark Rothko", ["Color Field Painting"], ["Abstract Painting"]),
    ("Willem de Kooning", ["Abstract Expressionism", "Action Painting"], ["Abstract Painting"]),
    ("Andy Warhol", ["Pop Art"], ["Portrait", "Illustration"]),
    ("Roy Lichtenstein", ["Pop Art"], ["Illustration", "Genre Painting"]),
    ("Salvador Dalí", ["Symbolism"], ["Landscape", "Genre Painting"]),
    ("René Magritte", ["Symbolism"], ["Genre Painting", "Portrait"]),
    ("Frida Kahlo", ["Naive Art Primitivism"], ["Portrait"]),
    ("Henri Rousseau", ["Naive Art Primitivism"], ["Landscape", "Genre Painting"]),
    ("Katsushika Hokusai", ["Ukiyo e"], ["Landscape"]),
    ("Utagawa Hiroshige", ["Ukiyo e"], ["Landscape", "Cityscape"]),
    ("Dante Gabriel Rossetti", ["Symbolism"], ["Portrait"]),
    ("Edward Hopper", ["Contemporary Realism", "Realism"], ["Cityscape", "Genre Painting"]),
]


def estimate_artist_from_style(
    period_top_k: list[tuple[str, float]],
    genre_top_k: list[tuple[str, float]],
    top_n: int = 5,
) -> list[CandidateAttribution]:
    """Rank candidate artists by how well they match the style prediction.

    Cross-references period and genre softmax outputs against known
    artist profiles.  This is a heuristic fallback — not a trained
    classifier — and should be presented with appropriate caveats.
    """
    period_scores: dict[str, float] = {label: conf for label, conf in period_top_k}
    genre_scores: dict[str, float] = {label: conf for label, conf in genre_top_k}

    artist_scores: list[tuple[str, float]] = []
    for artist, periods, genres in _ARTIST_PROFILES:
        period_score = max(
            (period_scores.get(p, 0.0) for p in periods), default=0.0,
        )
        genre_score = max(
            (genre_scores.get(g, 0.0) for g in genres), default=0.0,
        )
        combined = 0.65 * period_score + 0.35 * genre_score
        artist_scores.append((artist, combined))

    artist_scores.sort(key=lambda x: x[1], reverse=True)
    top = artist_scores[:top_n]

    total = sum(s for _, s in top) or 1.0
    return [
        CandidateAttribution(
            artist=artist,
            score=score / total,
            confidence_interval=(
                max(0.0, score / total - 0.15),
                min(1.0, score / total + 0.15),
            ),
            supporting_features=[
                "Estimated from style classification (period + genre)."
            ],
        )
        for artist, score in top
    ]
