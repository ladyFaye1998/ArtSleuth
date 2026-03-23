"""
WikiArt benchmark loader and evaluation.

Downloads and evaluates ArtSleuth on the WikiArt dataset (Saleh &
Elgammal, 2015) via HuggingFace Datasets.  The dataset contains
~81k artwork images with style, genre, and artist labels — the
standard benchmark for computational art analysis.

References
----------
Saleh, B. & Elgammal, A. (2015). Large-scale Classification of
    Fine-Art Paintings. *ICCV Workshop*.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# --- Data Structures --------------------------------------------------------


@dataclass
class BenchmarkSplit:
    """Metadata for a single evaluation split."""

    name: str
    num_samples: int
    num_classes: int
    class_names: list[str] = field(default_factory=list)


@dataclass
class ClassificationMetrics:
    """Evaluation metrics for a classification task."""

    accuracy: float = 0.0
    top5_accuracy: float = 0.0
    macro_f1: float = 0.0
    per_class_f1: dict[str, float] = field(default_factory=dict)
    confusion_matrix: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise metrics, excluding the confusion matrix."""
        d: dict[str, Any] = {
            "accuracy": round(self.accuracy, 4),
            "top5_accuracy": round(self.top5_accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
        }
        if self.per_class_f1:
            d["per_class_f1"] = {
                k: round(v, 4) for k, v in self.per_class_f1.items()
            }
        return d


@dataclass
class BenchmarkResult:
    """Complete benchmark evaluation for one backbone configuration."""

    backbone: str
    style_metrics: ClassificationMetrics | None = None
    artist_metrics: ClassificationMetrics | None = None
    genre_metrics: ClassificationMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON export."""
        out: dict[str, Any] = {"backbone": self.backbone}
        if self.style_metrics:
            out["style"] = self.style_metrics.to_dict()
        if self.artist_metrics:
            out["artist"] = self.artist_metrics.to_dict()
        if self.genre_metrics:
            out["genre"] = self.genre_metrics.to_dict()
        return out


# --- Dataset Loader ---------------------------------------------------------


def load_wikiart(
    *,
    split: str = "train",
    max_samples: int | None = None,
    cache_dir: Path | None = None,
) -> Any:
    """Load the WikiArt dataset from HuggingFace.

    Parameters
    ----------
    split:
        Dataset split (``"train"`` or ``"test"``).
    max_samples:
        Cap the number of samples for faster iteration.
    cache_dir:
        HuggingFace cache directory.

    Returns
    -------
    datasets.Dataset
        HuggingFace dataset with ``image``, ``style``, ``artist``,
        and ``genre`` columns.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the `datasets` package to use WikiArt benchmarks: "
            "pip install datasets"
        ) from exc

    logger.info("Loading WikiArt split=%s …", split)
    ds = load_dataset(
        "huggan/wikiart",
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    if max_samples is not None and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
        logger.info("Capped dataset to %d samples.", max_samples)

    return ds


# --- Feature Extraction -----------------------------------------------------


def extract_embeddings(
    dataset: Any,
    *,
    backbone: str = "dinov2",
    device: str = "cpu",
    batch_size: int = 32,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Extract backbone embeddings for all images in a dataset.

    Parameters
    ----------
    dataset:
        HuggingFace dataset with an ``image`` column.
    backbone:
        ``"dinov2"``, ``"clip"``, or ``"fusion"`` for the
        cross-attention backbone.
    device:
        PyTorch device.
    batch_size:
        Images per forward pass.
    cache_dir:
        Weight cache directory.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        ``(embeddings, labels)`` where ``embeddings`` has shape
        ``(N, D)`` and ``labels`` maps task names to integer arrays.
    """
    import torch
    from PIL import Image

    from artsleuth.config import BackboneType
    from artsleuth.models.backbones import load_backbone
    from artsleuth.preprocessing.transforms import prepare_for_backbone

    if backbone == "fusion":
        from artsleuth.models.fusion import DualBackboneFusion

        model = DualBackboneFusion(device=device, cache_dir=cache_dir)
        model.eval()
        bb_type = BackboneType.DINO_V2
    else:
        bb_type = BackboneType(backbone)
        model = load_backbone(bb_type, device=device, cache_dir=cache_dir)

    all_embeddings: list[np.ndarray] = []
    label_arrays: dict[str, list[int]] = {
        "style": [],
        "artist": [],
        "genre": [],
    }

    n = len(dataset)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_imgs = []
        for i in range(start, end):
            row = dataset[i]
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")
            tensor = prepare_for_backbone(img, bb_type)
            batch_imgs.append(tensor)

            for task in label_arrays:
                label_arrays[task].append(int(row.get(task, 0)))

        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = model(batch_tensor)

        all_embeddings.append(feats.cpu().numpy())

        if (start // batch_size) % 10 == 0:
            logger.info("Extracted %d / %d", end, n)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = {k: np.array(v) for k, v in label_arrays.items()}
    return embeddings, labels


# --- Linear Probe Evaluation ------------------------------------------------


def linear_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    class_names: list[str] | None = None,
) -> ClassificationMetrics:
    """Train a logistic-regression linear probe and evaluate.

    Parameters
    ----------
    train_embeddings:
        ``(N_train, D)`` feature matrix.
    train_labels:
        ``(N_train,)`` integer class labels.
    test_embeddings:
        ``(N_test, D)`` feature matrix.
    test_labels:
        ``(N_test,)`` integer class labels.
    class_names:
        Optional human-readable class names for per-class metrics.

    Returns
    -------
    ClassificationMetrics
        Accuracy, top-5 accuracy, macro F1, and per-class F1.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_embeddings)
    x_test = scaler.transform(test_embeddings)

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        C=1.0,
        n_jobs=-1,
    )
    clf.fit(x_train, train_labels)

    preds = clf.predict(x_test)
    probs = clf.predict_proba(x_test)

    acc = float(np.mean(preds == test_labels))

    top5 = _top_k_accuracy(probs, test_labels, k=5)
    macro = float(f1_score(test_labels, preds, average="macro", zero_division=0))

    per_class: dict[str, float] = {}
    if class_names:
        per_f1 = f1_score(test_labels, preds, average=None, zero_division=0)
        for i, name in enumerate(class_names):
            if i < len(per_f1):
                per_class[name] = float(per_f1[i])

    cm = confusion_matrix(test_labels, preds)

    return ClassificationMetrics(
        accuracy=acc,
        top5_accuracy=top5,
        macro_f1=macro,
        per_class_f1=per_class,
        confusion_matrix=cm,
    )


def _top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Compute top-k classification accuracy."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = np.any(top_k_preds == labels[:, None], axis=1)
    return float(np.mean(correct))


# --- Full Benchmark Runner --------------------------------------------------


def run_wikiart_benchmark(
    *,
    backbone: str = "dinov2",
    device: str = "cpu",
    max_samples: int | None = None,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> BenchmarkResult:
    """Run the complete WikiArt benchmark for a given backbone.

    Parameters
    ----------
    backbone:
        ``"dinov2"``, ``"clip"``, or ``"fusion"``.
    device:
        PyTorch device.
    max_samples:
        Cap per split for faster iteration.
    output_dir:
        Directory for JSON results.
    cache_dir:
        Weight and dataset cache directory.

    Returns
    -------
    BenchmarkResult
        Metrics across style, artist, and genre tasks.
    """
    logger.info("Running WikiArt benchmark: backbone=%s", backbone)

    train_ds = load_wikiart(split="train", max_samples=max_samples, cache_dir=cache_dir)
    test_ds = load_wikiart(split="test", max_samples=max_samples, cache_dir=cache_dir)

    logger.info("Extracting train embeddings …")
    train_emb, train_labels = extract_embeddings(
        train_ds, backbone=backbone, device=device, cache_dir=cache_dir,
    )
    logger.info("Extracting test embeddings …")
    test_emb, test_labels = extract_embeddings(
        test_ds, backbone=backbone, device=device, cache_dir=cache_dir,
    )

    result = BenchmarkResult(backbone=backbone)

    for task in ("style", "artist", "genre"):
        logger.info("Evaluating %s classification …", task)
        metrics = linear_probe(
            train_emb, train_labels[task],
            test_emb, test_labels[task],
        )
        setattr(result, f"{task}_metrics", metrics)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"wikiart_{backbone}.json"
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info("Results saved to %s", out_path)

    return result
