"""
Adversarial forgery robustness testing.

A forgery detector that only works on naive copies is of limited
practical value.  Historical forgers like Han van Meegeren and
Wolfgang Beltracchi employed sophisticated techniques — artificial
aging, period-appropriate materials, deliberate stylistic mimicry —
that fooled experts for decades.

This module simulates known forgery strategies as differentiable image
transforms and evaluates how robustly the ArtSleuth forgery detector
identifies the resulting anomalies.  The goal is not to create
forgeries but to stress-test detection under realistic adversarial
conditions.

References
----------
Lopez, O. et al. (2023). Adversarial Robustness in AI Art Authentication.
Wynne, F. (2006). *I Was Vermeer: The Rise and Fall of the Twentieth
    Century's Greatest Forger*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# --- Data Structures --------------------------------------------------------


@dataclass(frozen=True)
class ForgeryTechnique:
    """Descriptor for a single adversarial forgery technique.

    Attributes
    ----------
    name:
        Machine-readable identifier, e.g. ``"van_meegeren_aging"``.
    description:
        Art-historical context explaining the real-world technique
        this transform approximates.
    severity:
        How aggressively the transform is applied (0–1).
    """

    name: str
    description: str
    severity: float


@dataclass(frozen=True)
class AttackResult:
    """Outcome of running one forgery technique against the detector.

    Attributes
    ----------
    technique:
        The technique that was applied.
    original_anomaly_score:
        Detector's anomaly score on the clean image.
    adversarial_anomaly_score:
        Detector's anomaly score on the attacked image.
    detected:
        Whether the attacked image was still flagged as anomalous.
    score_delta:
        Change in anomaly score (negative means the detector was
        partially or fully fooled).
    """

    technique: ForgeryTechnique
    original_anomaly_score: float
    adversarial_anomaly_score: float
    detected: bool
    score_delta: float


@dataclass(frozen=True)
class RobustnessReport:
    """Aggregate report across all tested attack scenarios.

    Attributes
    ----------
    technique_results:
        Per-attack results.
    overall_detection_rate:
        Fraction of attacks that the detector still flagged.
    most_effective_attack:
        Technique name that reduced the anomaly score the most.
    mean_score_delta:
        Average score change across all attacks.
    """

    technique_results: list[AttackResult]
    overall_detection_rate: float
    most_effective_attack: str
    mean_score_delta: float


# --- Forgery Simulator ------------------------------------------------------


_TECHNIQUE_DESCRIPTORS: dict[str, tuple[str, str]] = {
    "van_meegeren_aging": (
        "artificial_aging",
        "Simulates Han van Meegeren's artificial aging process: baking "
        "canvases to induce craquelure, applying tinted varnish to mimic "
        "centuries of oxidation, and abrading the surface to suggest wear.",
    ),
    "beltracchi_style_transfer": (
        "style_transfer_perturbation",
        "Approximates Wolfgang Beltracchi's method of absorbing a painter's "
        "palette and compositional habits so thoroughly that the resulting "
        "work reads as a plausible 'lost' piece from the target period.",
    ),
    "material_anachronism": (
        "material_anachronism",
        "Introduces subtle material inconsistencies — the kind that "
        "spectrographic or X-ray fluorescence analysis eventually reveals: "
        "modern pigment distributions, uniform weave textures foreign to "
        "historical canvases.",
    ),
    "composite_forgery": (
        "composite_forgery",
        "Combines aging, style perturbation, and material anachronism into "
        "a single layered attack, mirroring the most sophisticated forgers "
        "who addressed surface, style, and materials simultaneously.",
    ),
}


class ForgerySimulator:
    """Generates adversarial test images from known historical forgery techniques.

    Each method applies a deterministic (given ``random_state``) image
    transform inspired by a documented forgery strategy.  The transforms
    operate on PIL Images via NumPy — no GPU or neural model required.

    Parameters
    ----------
    random_state:
        Seed for reproducible perturbations.
    """

    def __init__(self, *, random_state: int = 42) -> None:
        self._rng = np.random.RandomState(random_state)

    # --- Public Techniques --------------------------------------------------

    def artificial_aging(
        self, image: Image.Image, severity: float = 0.5
    ) -> Image.Image:
        """Simulate centuries of aging on a painting surface.

        Applies three layered effects whose intensity scales with
        *severity*:

        1. Yellow-brown varnish overlay (adjustable opacity).
        2. Simulated craquelure via random Voronoi-like dark lines.
        3. Slight Gaussian blur to suggest surface wear.

        Parameters
        ----------
        image:
            Input RGB image.
        severity:
            Intensity of the aging effect (0–1).

        Returns
        -------
        Image.Image
            Aged version of the input.
        """
        severity = float(np.clip(severity, 0.0, 1.0))
        arr = np.array(image, dtype=np.float64) / 255.0

        # -- Varnish overlay: warm yellow-brown tint
        varnish_color = np.array([0.85, 0.72, 0.45])
        varnish_alpha = 0.15 + 0.35 * severity
        arr = arr * (1.0 - varnish_alpha) + varnish_color * varnish_alpha

        # -- Simulated craquelure
        arr = self._add_craquelure(arr, severity)

        # -- Surface wear blur
        blur_sigma = 0.3 + 1.2 * severity
        for ch in range(arr.shape[2]):
            arr[:, :, ch] = gaussian_filter(arr[:, :, ch], sigma=blur_sigma)

        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def style_transfer_perturbation(
        self, image: Image.Image, severity: float = 0.5
    ) -> Image.Image:
        """Approximate neural style transfer via histogram and frequency manipulation.

        Two complementary effects:

        1. **Histogram shifting** — redistributes each colour channel
           toward a target period palette (warm, desaturated).
        2. **Frequency-domain texture perturbation** — boosts low
           frequencies and attenuates high frequencies to soften
           characteristic brushwork signatures.

        Parameters
        ----------
        image:
            Input RGB image.
        severity:
            Intensity of the perturbation (0–1).

        Returns
        -------
        Image.Image
            Style-perturbed version of the input.
        """
        severity = float(np.clip(severity, 0.0, 1.0))
        arr = np.array(image, dtype=np.float64) / 255.0

        # -- Histogram shift toward a warm, desaturated target palette
        target_means = np.array([0.55, 0.45, 0.35])
        target_stds = np.array([0.18, 0.16, 0.14])
        blend = 0.2 + 0.5 * severity

        for ch in range(3):
            ch_data = arr[:, :, ch]
            src_mean = ch_data.mean()
            src_std = ch_data.std() + 1e-12
            shifted = (ch_data - src_mean) / src_std * target_stds[ch] + target_means[ch]
            arr[:, :, ch] = ch_data * (1.0 - blend) + shifted * blend

        # -- Frequency-domain texture manipulation
        arr = self._perturb_frequencies(arr, severity)

        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def material_anachronism(
        self, image: Image.Image, severity: float = 0.5
    ) -> Image.Image:
        """Inject subtle modern-material signatures into an image.

        Two effects that mimic what spectrographic analysis might catch:

        1. **Spectral signature injection** — slight shifts in colour
           channel distributions that diverge from period-appropriate
           pigment spectra.
        2. **Uniform texture overlay** — a faint regular pattern
           (simulating machine-woven canvas) foreign to hand-prepared
           historical supports.

        Parameters
        ----------
        image:
            Input RGB image.
        severity:
            Intensity of the anachronism (0–1).

        Returns
        -------
        Image.Image
            Image with injected material anomalies.
        """
        severity = float(np.clip(severity, 0.0, 1.0))
        arr = np.array(image, dtype=np.float64) / 255.0

        # -- Spectral signature: boost blue channel, suppress red slightly
        spectral_shift = np.zeros_like(arr)
        spectral_shift[:, :, 0] = -0.03 * severity
        spectral_shift[:, :, 2] = 0.05 * severity
        arr = arr + spectral_shift

        # -- Uniform weave texture (periodic grid)
        h, w = arr.shape[:2]
        y_coords = np.arange(h)
        x_coords = np.arange(w)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

        period = max(4, int(12 - 6 * severity))
        weave = (
            np.sin(2.0 * np.pi * yy / period)
            * np.sin(2.0 * np.pi * xx / period)
        )
        weave_intensity = 0.01 + 0.04 * severity
        for ch in range(3):
            arr[:, :, ch] = arr[:, :, ch] + weave * weave_intensity

        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def composite_forgery(
        self, image: Image.Image, severity: float = 0.5
    ) -> Image.Image:
        """Layer all three forgery techniques into a single attack.

        Applies aging, style perturbation, and material anachronism
        sequentially, each at the requested severity.  This mirrors the
        most sophisticated forgers who addressed surface appearance,
        stylistic fingerprint, and material provenance simultaneously.

        Parameters
        ----------
        image:
            Input RGB image.
        severity:
            Intensity of the composite attack (0–1).

        Returns
        -------
        Image.Image
            Compositely attacked image.
        """
        result = self.artificial_aging(image, severity)
        result = self.style_transfer_perturbation(result, severity)
        result = self.material_anachronism(result, severity)
        return result

    def available_techniques(self) -> list[ForgeryTechnique]:
        """Return descriptors for every registered forgery technique.

        Returns
        -------
        list[ForgeryTechnique]
            One entry per technique, with default severity of 0.5.
        """
        techniques: list[ForgeryTechnique] = []
        for name, (_method_name, description) in _TECHNIQUE_DESCRIPTORS.items():
            techniques.append(
                ForgeryTechnique(name=name, description=description, severity=0.5)
            )
        return techniques

    # --- Internal Helpers ---------------------------------------------------

    def _add_craquelure(
        self, arr: np.ndarray, severity: float
    ) -> np.ndarray:
        """Draw random Voronoi-like dark lines to simulate paint cracking."""
        h, w = arr.shape[:2]
        n_seeds = int(20 + 80 * severity)

        seed_y = self._rng.randint(0, h, size=n_seeds)
        seed_x = self._rng.randint(0, w, size=n_seeds)

        # Build a rough Voronoi boundary map via a distance-label approach:
        # assign each pixel to its nearest seed, then mark pixels whose
        # neighbours belong to a different region.
        yy, xx = np.mgrid[:h, :w]
        distances = np.full((h, w), np.inf, dtype=np.float64)
        labels = np.zeros((h, w), dtype=np.int32)

        for i in range(n_seeds):
            d = (yy - seed_y[i]) ** 2 + (xx - seed_x[i]) ** 2
            closer = d < distances
            distances[closer] = d[closer]
            labels[closer] = i

        # Detect boundaries: pixels adjacent to a different label
        boundary = np.zeros((h, w), dtype=bool)
        boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
        boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]

        # Darken boundary pixels
        crack_darkness = 0.3 + 0.4 * severity
        crack_mask = boundary.astype(np.float64) * crack_darkness
        crack_mask = gaussian_filter(crack_mask, sigma=0.6)

        for ch in range(arr.shape[2]):
            arr[:, :, ch] = arr[:, :, ch] * (1.0 - crack_mask)

        return arr

    def _perturb_frequencies(
        self, arr: np.ndarray, severity: float
    ) -> np.ndarray:
        """Manipulate frequency bands to obscure brushwork texture.

        Boosts low frequencies (broad tonal structure) and attenuates
        high frequencies (fine texture detail) in the spatial-frequency
        domain via the 2-D DFT.
        """
        h, w = arr.shape[:2]
        result = arr.copy()

        # Build radial frequency mask
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[:h, :w]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_radius = np.sqrt(cy ** 2 + cx ** 2) + 1e-12

        # Normalise radius to [0, 1]
        r_norm = radius / max_radius

        # Low-pass boost, high-frequency attenuation
        boost_factor = 1.0 + 0.3 * severity
        attenuate_factor = 1.0 - 0.5 * severity
        freq_weight = np.where(
            r_norm < 0.3,
            boost_factor,
            np.where(r_norm > 0.7, attenuate_factor, 1.0),
        )

        # Smooth transition between regions
        freq_weight = gaussian_filter(freq_weight, sigma=5.0)

        for ch in range(3):
            f = np.fft.fft2(arr[:, :, ch])
            f_shifted = np.fft.fftshift(f)
            f_shifted *= freq_weight
            result[:, :, ch] = np.real(np.fft.ifft2(np.fft.ifftshift(f_shifted)))

        return result

    def _apply_technique(
        self, name: str, image: Image.Image, severity: float
    ) -> Image.Image:
        """Dispatch to the named technique method.

        Parameters
        ----------
        name:
            Technique identifier (must match a key in
            ``_TECHNIQUE_DESCRIPTORS``).
        image:
            Input RGB image.
        severity:
            Transform intensity (0–1).

        Returns
        -------
        Image.Image
            Transformed image.

        Raises
        ------
        ValueError
            If *name* is not a recognised technique.
        """
        method_map: dict[str, str] = {
            k: v[0] for k, v in _TECHNIQUE_DESCRIPTORS.items()
        }
        method_name = method_map.get(name)
        if method_name is None:
            known = ", ".join(sorted(method_map))
            raise ValueError(
                f"Unknown technique {name!r}. Available: {known}"
            )
        method = getattr(self, method_name)
        return method(image, severity)


# --- Robustness Evaluator ---------------------------------------------------


class RobustnessEvaluator:
    """Evaluate forgery-detector robustness under adversarial conditions.

    Pairs a :class:`ForgerySimulator` with a
    :class:`~artsleuth.core.forgery.ForgeryDetector` and systematically
    measures how well the detector identifies anomalies after each
    simulated forgery technique has been applied.

    Parameters
    ----------
    detector:
        A ``ForgeryDetector`` instance (typed as ``Any`` to avoid
        circular imports).
    simulator:
        Optional pre-configured simulator; a default instance is
        created when ``None``.
    """

    def __init__(
        self,
        detector: Any,
        simulator: ForgerySimulator | None = None,
    ) -> None:
        self._detector = detector
        self._simulator = simulator or ForgerySimulator()

    # --- Public API ---------------------------------------------------------

    def evaluate(
        self,
        image: Image.Image,
        reference_artist: str,
        *,
        techniques: list[str] | None = None,
        severities: list[float] | None = None,
    ) -> RobustnessReport:
        """Run a full robustness evaluation across techniques and severities.

        Parameters
        ----------
        image:
            Clean RGB artwork image to attack.
        reference_artist:
            Artist whose reference corpus the detector uses.
        techniques:
            Technique names to test.  ``None`` tests all available.
        severities:
            Severity levels to sweep.  ``None`` defaults to
            ``[0.3, 0.5, 0.7]``.

        Returns
        -------
        RobustnessReport
            Aggregated results across every technique x severity
            combination.
        """
        if techniques is None:
            techniques = [
                t.name for t in self._simulator.available_techniques()
            ]
        if severities is None:
            severities = [0.3, 0.5, 0.7]

        results: list[AttackResult] = []
        for tech_name in techniques:
            for sev in severities:
                result = self.evaluate_single(
                    image, reference_artist, tech_name, sev
                )
                results.append(result)

        return self._aggregate(results)

    def evaluate_single(
        self,
        image: Image.Image,
        reference_artist: str,
        technique: str,
        severity: float = 0.5,
    ) -> AttackResult:
        """Test one technique at one severity level.

        Parameters
        ----------
        image:
            Clean RGB artwork image.
        reference_artist:
            Artist whose reference corpus the detector uses.
        technique:
            Technique identifier.
        severity:
            Transform intensity (0–1).

        Returns
        -------
        AttackResult
            Scores before and after the attack, and whether detection
            survived.
        """
        original_report = self._detector.detect(image, reference_artist)
        original_score = original_report.anomaly_score

        adversarial_image = self._simulator._apply_technique(
            technique, image, severity
        )
        adversarial_report = self._detector.detect(
            adversarial_image, reference_artist
        )
        adversarial_score = adversarial_report.anomaly_score

        description = _TECHNIQUE_DESCRIPTORS.get(technique, ("", ""))[1]
        tech = ForgeryTechnique(
            name=technique, description=description, severity=severity
        )

        return AttackResult(
            technique=tech,
            original_anomaly_score=original_score,
            adversarial_anomaly_score=adversarial_score,
            detected=adversarial_report.is_flagged,
            score_delta=adversarial_score - original_score,
        )

    # --- Internal Helpers ---------------------------------------------------

    @staticmethod
    def _aggregate(results: list[AttackResult]) -> RobustnessReport:
        """Collapse per-attack results into a summary report."""
        if not results:
            return RobustnessReport(
                technique_results=[],
                overall_detection_rate=0.0,
                most_effective_attack="none",
                mean_score_delta=0.0,
            )

        detected_count = sum(1 for r in results if r.detected)
        detection_rate = detected_count / len(results)

        deltas = [r.score_delta for r in results]
        mean_delta = float(np.mean(deltas))

        most_effective = min(results, key=lambda r: r.score_delta)

        return RobustnessReport(
            technique_results=results,
            overall_detection_rate=detection_rate,
            most_effective_attack=most_effective.technique.name,
            mean_score_delta=mean_delta,
        )
