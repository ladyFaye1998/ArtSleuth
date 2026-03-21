"""
Brushstroke pattern extraction and analysis.

This module implements the computational analogue of what connoisseurs
call *facture* — the characteristic handling of paint that distinguishes
one artist's hand from another.  By decomposing a painting into local
patches and projecting each through a self-supervised vision transformer,
we obtain dense feature maps that encode directional energy, texture
granularity, and impasto relief.

The approach draws on the observation — formalised by Morelli in the
1870s and refined by Berenson — that an artist's most habitual gestures
reside in the least-scrutinised passages: drapery folds, background
foliage, the rendering of fingernails and earlobes.

References
----------
Morelli, G. (1890). *Italian Painters: Critical Studies of Their Works*.
Berenson, B. (1902). *The Study and Criticism of Italian Art*.
Caron, M. et al. (2021). Emerging Properties in Self-Supervised Vision
    Transformers. *ICCV*.
Oquab, M. et al. (2024). DINOv2: Learning Robust Visual Features
    without Supervision. *TMLR*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL import Image

    from artsleuth.config import AnalysisConfig


# --- Data Structures --------------------------------------------------------


@dataclass(frozen=True)
class StrokeDescriptor:
    """Quantitative description of a single brushstroke region.

    Attributes
    ----------
    orientation:
        Dominant stroke angle in radians (0 = horizontal, π/2 = vertical).
    coherence:
        Directional consistency within the patch (0 = isotropic, 1 = perfectly aligned).
    energy:
        Mean gradient magnitude, a proxy for impasto thickness.
    curvature:
        Estimated stroke curvature derived from the structure tensor.
    embedding:
        High-dimensional feature vector from the vision-transformer backbone.
    bbox:
        Patch bounding box as (x, y, width, height) in pixel coordinates.
    """

    orientation: float
    coherence: float
    energy: float
    curvature: float
    embedding: np.ndarray
    bbox: tuple[int, int, int, int]


@dataclass
class BrushstrokeReport:
    """Aggregated brushstroke analysis for a complete artwork.

    Attributes
    ----------
    descriptors:
        Per-patch stroke descriptors across the painting surface.
    mean_coherence:
        Average directional consistency — higher values suggest a more
        disciplined or systematic hand.
    orientation_histogram:
        Binned distribution of dominant stroke angles (36 bins, 10° each).
    energy_map:
        Spatial heatmap of gradient energy at the original image resolution.
    cluster_labels:
        Cluster assignment for each patch, grouping regions with similar
        stroke characteristics (useful for detecting multiple hands in a
        workshop painting).
    """

    descriptors: list[StrokeDescriptor] = field(default_factory=list)
    mean_coherence: float = 0.0
    orientation_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(36, dtype=np.float32)
    )
    energy_map: np.ndarray | None = None
    cluster_labels: np.ndarray | None = None


# --- Analyzer ---------------------------------------------------------------


class BrushstrokeAnalyzer:
    """Extracts and clusters brushstroke patterns from artwork images.

    The analyzer operates in three stages:

    1. **Patch extraction** — The input image is divided into overlapping
       patches according to the configured strategy.
    2. **Feature computation** — Each patch is passed through a pre-trained
       vision transformer to obtain a dense feature map, from which
       orientation, coherence, energy, and curvature are derived via the
       structure tensor.
    3. **Clustering** — Patch embeddings are clustered to reveal regions
       of stylistically homogeneous brushwork, potentially corresponding
       to distinct hands in a workshop context.

    Parameters
    ----------
    config:
        Analysis configuration governing patch size, backbone choice,
        and device placement.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config
        self._device = config.resolve_device()
        self._backbone: torch.nn.Module | None = None

    # --- Public API ---------------------------------------------------------

    def analyze(self, image: Image.Image) -> BrushstrokeReport:
        """Run full brushstroke analysis on a PIL image.

        Parameters
        ----------
        image:
            RGB artwork image (any resolution; will be internally resized
            if it exceeds ``config.max_resolution``).

        Returns
        -------
        BrushstrokeReport
            Aggregated stroke descriptors, statistics, and cluster labels.
        """
        patches, bboxes = self._extract_patches(image)
        descriptors = self._compute_descriptors(patches, bboxes)
        report = self._aggregate(descriptors, image.size)
        return report

    # --- Internal Methods ---------------------------------------------------

    def _ensure_backbone(self) -> torch.nn.Module:
        """Lazy-load the vision-transformer backbone."""
        if self._backbone is None:
            from artsleuth.models.backbones import load_backbone

            self._backbone = load_backbone(
                self._config.backbone,
                device=self._device,
                cache_dir=self._config.cache_dir,
            )
        return self._backbone

    def _extract_patches(
        self, image: Image.Image
    ) -> tuple[list[torch.Tensor], list[tuple[int, int, int, int]]]:
        """Tile the artwork into analysis patches.

        Delegates to the preprocessing module to respect the configured
        patch strategy (grid, salient, or adaptive).
        """
        from artsleuth.preprocessing.patches import extract_patches

        return extract_patches(
            image,
            patch_size=self._config.patch_size,
            strategy=self._config.patch_strategy,
            max_resolution=self._config.max_resolution,
        )

    def _compute_descriptors(
        self,
        patches: list[torch.Tensor],
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[StrokeDescriptor]:
        """Compute stroke descriptors for each patch via the backbone."""
        backbone = self._ensure_backbone()
        descriptors: list[StrokeDescriptor] = []

        batch = torch.stack(patches).to(self._device)

        with torch.no_grad():
            features = backbone(batch)

        for i, (feat, bbox) in enumerate(zip(features, bboxes)):
            feat_np = feat.cpu().numpy()
            orientation, coherence, energy, curvature = self._structure_tensor_stats(
                patches[i]
            )

            descriptors.append(
                StrokeDescriptor(
                    orientation=orientation,
                    coherence=coherence,
                    energy=energy,
                    curvature=curvature,
                    embedding=feat_np.flatten(),
                    bbox=bbox,
                )
            )

        return descriptors

    @staticmethod
    def _structure_tensor_stats(
        patch: torch.Tensor,
    ) -> tuple[float, float, float, float]:
        """Derive orientation and coherence from the 2D structure tensor.

        The structure tensor captures the local distribution of gradient
        directions.  Its eigenvalues (λ₁ ≥ λ₂) encode:
          - Orientation:  angle of the dominant eigenvector.
          - Coherence:    (λ₁ − λ₂) / (λ₁ + λ₂ + ε), ranging 0 → 1.
          - Energy:       √(λ₁ + λ₂), proportional to gradient magnitude.
          - Curvature:    estimated via the Hessian eigenvalue ratio.
        """
        grey = patch.mean(dim=0).numpy() if patch.dim() == 3 else patch.numpy()

        gy, gx = np.gradient(grey.astype(np.float64))

        jxx = (gx * gx).mean()
        jxy = (gx * gy).mean()
        jyy = (gy * gy).mean()

        # Eigenvalues of the 2×2 structure tensor
        discriminant = np.sqrt(max((jxx - jyy) ** 2 + 4 * jxy**2, 0.0))
        lambda1 = 0.5 * (jxx + jyy + discriminant)
        lambda2 = 0.5 * (jxx + jyy - discriminant)

        orientation = 0.5 * np.arctan2(2 * jxy, jxx - jyy + 1e-12)
        coherence = float((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-12))
        energy = float(np.sqrt(lambda1 + lambda2))

        # Curvature from Hessian trace ratio
        hyy, _ = np.gradient(gy)
        _, hxx = np.gradient(gx)
        trace = float(np.abs(hyy + hxx).mean())
        curvature = trace / (energy + 1e-12)

        return float(orientation), coherence, energy, curvature

    def _aggregate(
        self,
        descriptors: list[StrokeDescriptor],
        image_size: tuple[int, int],
    ) -> BrushstrokeReport:
        """Aggregate per-patch descriptors into a full-painting report."""
        from sklearn.cluster import KMeans

        if not descriptors:
            return BrushstrokeReport()

        # Orientation histogram (36 bins × 10°)
        orientations = np.array([d.orientation for d in descriptors])
        hist, _ = np.histogram(
            orientations, bins=36, range=(-np.pi / 2, np.pi / 2), density=True
        )

        mean_coherence = float(np.mean([d.coherence for d in descriptors]))

        # Energy heatmap projected onto image grid
        w, h = image_size
        energy_map = np.zeros((h, w), dtype=np.float32)
        for d in descriptors:
            x, y, pw, ph = d.bbox
            energy_map[y : y + ph, x : x + pw] = d.energy

        # Cluster patch embeddings to reveal distinct hands
        embeddings = np.stack([d.embedding for d in descriptors])
        n_clusters = min(4, len(descriptors))
        if n_clusters >= 2:
            labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(
                embeddings
            )
        else:
            labels = np.zeros(len(descriptors), dtype=np.int32)

        return BrushstrokeReport(
            descriptors=descriptors,
            mean_coherence=mean_coherence,
            orientation_histogram=hist.astype(np.float32),
            energy_map=energy_map,
            cluster_labels=labels,
        )
