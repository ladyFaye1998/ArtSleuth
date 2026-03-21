"""
Hierarchical workshop decomposition.

Renaissance and Baroque workshops were small factories: a master
designed the composition, painted the principal figures, and delegated
secondary passages to assistants.  Identifying which patches belong to
which hand is central to attribution — and to understanding how a
painting was actually made.

Flat k-means clustering treats all hands as equal.  This module uses a
variational Bayesian Gaussian mixture model with a Dirichlet process
prior, which automatically infers the number of distinct hands and
assigns a probabilistic confidence to each patch.  The largest cluster
is labelled "primary hand" (typically the master); smaller clusters
are labelled as secondary hands.

References
----------
Blei, D. M. & Jordan, M. I. (2006). Variational Inference for
    Dirichlet Process Mixtures. *Bayesian Analysis*, 1(1), 121–143.
Ainsworth, M. W. (2005). From Connoisseurship to Technical Art History.
    *Getty Research Journal*, 159–176.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sklearn.mixture import BayesianGaussianMixture


# --- Data Structures --------------------------------------------------------


@dataclass(frozen=True)
class HandAssignment:
    """Description of a single inferred hand within a workshop painting.

    Attributes
    ----------
    hand_id:
        Cluster index produced by the mixture model.
    label:
        Human-readable label: ``"primary_hand"`` for the dominant
        cluster, ``"secondary_hand_1"``, ``"secondary_hand_2"``, etc.
        for the rest.
    confidence:
        Mean posterior probability of this assignment across the
        patches attributed to this hand.
    patch_count:
        Number of patches assigned to this hand.
    spatial_extent:
        Fraction of the total image area covered by this hand's patches.
    mean_coherence:
        Average brushstroke coherence within this hand's patches.
        ``0.0`` when coherence data is unavailable.
    mean_energy:
        Average brushstroke energy within this hand's patches.
        ``0.0`` when energy data is unavailable.
    """

    hand_id: int
    label: str
    confidence: float
    patch_count: int
    spatial_extent: float
    mean_coherence: float
    mean_energy: float


@dataclass(frozen=True)
class WorkshopReport:
    """Complete workshop decomposition for an artwork.

    Attributes
    ----------
    num_hands:
        Inferred number of distinct hands (after pruning negligible
        components).
    is_workshop:
        ``True`` when more than one hand is detected with sufficient
        confidence — a strong signal of workshop production.
    assignments:
        One :class:`HandAssignment` per detected hand, sorted by
        ``patch_count`` descending (primary hand first).
    patch_labels:
        Integer array of length *n_patches* giving the hand assignment
        for every input patch.
    patch_probabilities:
        ``(n_patches, num_hands)`` matrix of posterior probabilities.
    hand_map:
        Spatial heatmap at original image resolution where each pixel
        carries the hand-id of the patch that covers it.  ``None`` when
        bounding boxes are unavailable.
    bic_score:
        BIC-like model-selection score for diagnostics (lower is
        better).
    """

    num_hands: int
    is_workshop: bool
    assignments: list[HandAssignment]
    patch_labels: np.ndarray
    patch_probabilities: np.ndarray
    hand_map: np.ndarray | None
    bic_score: float


# --- Decomposition Engine ---------------------------------------------------


class WorkshopDecomposition:
    """Bayesian workshop decomposition via a Dirichlet-process mixture.

    Instead of forcing a fixed *k* (as k-means does), a variational
    Bayesian Gaussian mixture with a Dirichlet-process prior lets the
    data decide how many distinct hands are present.  Components that
    capture fewer than ``min_hand_fraction`` of the patches are pruned
    as noise, and the remaining clusters are ranked by size.

    Parameters
    ----------
    max_hands:
        Upper bound on the number of mixture components.  The Dirichlet
        process will use fewer if the data warrant it.
    min_hand_fraction:
        Minimum fraction of total patches a component must capture to
        be retained as a genuine hand.  Filters out noise clusters.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        *,
        max_hands: int = 6,
        min_hand_fraction: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self._max_hands = max_hands
        self._min_hand_fraction = min_hand_fraction
        self._random_state = random_state

    # --- Public API ---------------------------------------------------------

    def decompose(
        self,
        embeddings: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
        image_size: tuple[int, int],
        *,
        coherences: np.ndarray | None = None,
        energies: np.ndarray | None = None,
    ) -> WorkshopReport:
        """Run hierarchical workshop decomposition on patch embeddings.

        Parameters
        ----------
        embeddings:
            ``(n_patches, dim)`` feature matrix — typically 384-d from
            DINOv2 or 512-d from a fusion head.
        bboxes:
            Per-patch bounding boxes as ``(x, y, width, height)`` in
            pixel coordinates.
        image_size:
            ``(width, height)`` of the original artwork image.
        coherences:
            Optional per-patch scalar array of brushstroke coherence
            values.  Used only to enrich :class:`HandAssignment`
            statistics; does not affect clustering.
        energies:
            Optional per-patch scalar array of brushstroke energy
            values.  Same role as *coherences*.

        Returns
        -------
        WorkshopReport
            Full decomposition including per-patch labels, posterior
            probabilities, spatial hand map, and diagnostics.
        """
        from sklearn.mixture import BayesianGaussianMixture

        n_patches = embeddings.shape[0]

        model = BayesianGaussianMixture(
            n_components=self._max_hands,
            covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            random_state=self._random_state,
            max_iter=300,
        )
        model.fit(embeddings)

        raw_labels = model.predict(embeddings)
        raw_probs = model.predict_proba(embeddings)

        # --- Prune negligible components ------------------------------------
        unique_labels, counts = np.unique(raw_labels, return_counts=True)
        fractions = counts / n_patches
        kept_mask = fractions >= self._min_hand_fraction
        kept_labels = unique_labels[kept_mask]

        if len(kept_labels) == 0:
            kept_labels = unique_labels[np.argmax(counts)].reshape(1)

        old_to_new: dict[int, int] = {}
        sorted_indices = np.argsort(
            [-counts[np.where(unique_labels == lbl)[0][0]] for lbl in kept_labels]
        )
        for new_id, sort_idx in enumerate(sorted_indices):
            old_to_new[int(kept_labels[sort_idx])] = new_id
        num_hands = len(old_to_new)

        # Remap patch labels; patches from pruned components get assigned
        # to their most-probable surviving component.
        patch_labels = np.empty(n_patches, dtype=np.int32)
        kept_cols = [
            int(np.where(unique_labels == old)[0][0])
            for old in kept_labels
        ]
        kept_probs = raw_probs[:, kept_cols]

        for i in range(n_patches):
            old_lbl = int(raw_labels[i])
            if old_lbl in old_to_new:
                patch_labels[i] = old_to_new[old_lbl]
            else:
                best_kept = int(np.argmax(kept_probs[i]))
                patch_labels[i] = old_to_new[int(kept_labels[best_kept])]

        # Build renormalized posterior matrix over surviving hands
        col_order = [
            int(np.where(unique_labels == old)[0][0])
            for old, _ in sorted(old_to_new.items(), key=lambda kv: kv[1])
        ]
        patch_probabilities = raw_probs[:, col_order]
        row_sums = patch_probabilities.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        patch_probabilities = patch_probabilities / row_sums

        # --- Compute total image area from bboxes ---------------------------
        total_area = float(sum(w * h for _, _, w, h in bboxes)) or 1.0

        # --- Build HandAssignment objects -----------------------------------
        assignments: list[HandAssignment] = []
        for new_id in range(num_hands):
            mask = patch_labels == new_id
            count = int(mask.sum())
            hand_area = float(sum(
                bboxes[j][2] * bboxes[j][3]
                for j in np.where(mask)[0]
            ))
            spatial_extent = hand_area / total_area

            confidence = float(patch_probabilities[mask, new_id].mean()) if count else 0.0

            mean_coh = 0.0
            if coherences is not None and count > 0:
                mean_coh = float(coherences[mask].mean())

            mean_ene = 0.0
            if energies is not None and count > 0:
                mean_ene = float(energies[mask].mean())

            label = "primary_hand" if new_id == 0 else f"secondary_hand_{new_id}"

            assignments.append(HandAssignment(
                hand_id=new_id,
                label=label,
                confidence=confidence,
                patch_count=count,
                spatial_extent=spatial_extent,
                mean_coherence=mean_coh,
                mean_energy=mean_ene,
            ))

        # --- Spatial hand map -----------------------------------------------
        hand_map: np.ndarray | None = None
        if bboxes:
            hand_map = self._build_hand_map(patch_labels, bboxes, image_size)

        bic_score = self._compute_bic(model, embeddings)

        is_workshop = num_hands > 1 and all(
            a.confidence > self._min_hand_fraction for a in assignments
        )

        return WorkshopReport(
            num_hands=num_hands,
            is_workshop=is_workshop,
            assignments=assignments,
            patch_labels=patch_labels,
            patch_probabilities=patch_probabilities,
            hand_map=hand_map,
            bic_score=bic_score,
        )

    # --- Internal Methods ---------------------------------------------------

    @staticmethod
    def _build_hand_map(
        labels: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Paint each patch's bounding box onto an image-resolution grid.

        Parameters
        ----------
        labels:
            Integer hand-id for every patch.
        bboxes:
            Per-patch ``(x, y, width, height)`` bounding boxes.
        image_size:
            ``(width, height)`` of the target canvas.

        Returns
        -------
        np.ndarray
            ``(height, width)`` array of dtype ``int32``, initialised to
            ``-1`` (background) and painted with hand-ids where patches
            fall.  Later patches overwrite earlier ones when they overlap.
        """
        w, h = image_size
        hand_map = np.full((h, w), fill_value=-1, dtype=np.int32)
        for idx, (bx, by, bw, bh) in enumerate(bboxes):
            x0 = max(bx, 0)
            y0 = max(by, 0)
            x1 = min(bx + bw, w)
            y1 = min(by + bh, h)
            if x1 > x0 and y1 > y0:
                hand_map[y0:y1, x0:x1] = int(labels[idx])
        return hand_map

    @staticmethod
    def _compute_bic(model: BayesianGaussianMixture, X: np.ndarray) -> float:
        """Approximate the Bayesian Information Criterion for the fitted model.

        Parameters
        ----------
        model:
            Fitted :class:`~sklearn.mixture.BayesianGaussianMixture`.
        X:
            ``(n_samples, n_features)`` data matrix used for fitting.

        Returns
        -------
        float
            BIC score (lower indicates a more parsimonious fit).

        Notes
        -----
        scikit-learn's ``BayesianGaussianMixture`` does not expose a
        ``bic`` method.  We approximate it as
        ``-2 * log_likelihood + k * ln(n)`` where *k* is the number of
        free parameters for the active components.
        """
        n_samples, n_features = X.shape
        log_likelihood = float(model.score(X)) * n_samples

        active = model.weights_ > 1e-6
        n_active = int(active.sum())

        # Per-component free parameters: mean (d) + full covariance (d*(d+1)/2)
        # + 1 mixing weight
        params_per_component = n_features + n_features * (n_features + 1) // 2 + 1
        n_params = n_active * params_per_component

        return -2.0 * log_likelihood + n_params * np.log(n_samples)
