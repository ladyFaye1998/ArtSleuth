"""
Publication-quality visualization utilities.

Gallery-warm colour palette with muted backgrounds, designed to produce
figures suitable for catalogue essays, conservation reports, and academic
publications without additional styling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

    from artsleuth.core.brushstroke import BrushstrokeReport
    from artsleuth.core.pipeline import AnalysisResult
    from artsleuth.core.style import StyleReport


# --- Colour Palette (gallery-warm) ------------------------------------------

PALETTE = {
    "background": "#f7f5f2",
    "ink": "#1A2E48",
    "rose": "#D4899A",
    "blue": "#9DC0D8",
    "gold": "#d4af37",
    "cream": "#faf8f5",
}


# --- Public API -------------------------------------------------------------


def save_heatmap_overlay(
    composite: np.ndarray,
    path: str | Path,
    *,
    dpi: int = 300,
) -> None:
    """Save a heatmap composite image to disk.

    Parameters
    ----------
    composite:
        RGB or RGBA numpy array.
    path:
        Output file path.
    dpi:
        Resolution for rasterised output.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=PALETTE["background"])
    ax.imshow(composite)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_style_distribution(
    style_report: StyleReport,
    path: str | Path,
    *,
    dpi: int = 300,
) -> None:
    """Plot a horizontal bar chart of style predictions.

    Parameters
    ----------
    style_report:
        Style classification report.
    path:
        Output file path.
    dpi:
        Output resolution.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=PALETTE["background"])

    colors = [PALETTE["rose"], PALETTE["blue"], PALETTE["gold"]]
    axis_data = [
        ("Period", style_report.period),
        ("School", style_report.school),
        ("Genre", style_report.technique),
    ]

    for ax, color, (title, pred) in zip(axes, colors, axis_data, strict=False):
        labels = [label for label, _ in pred.top_k]
        scores = [score for _, score in pred.top_k]

        ax.barh(labels[::-1], scores[::-1], color=color, edgecolor=PALETTE["ink"], linewidth=0.5)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=14, color=PALETTE["ink"], fontweight="bold")
        ax.tick_params(colors=PALETTE["ink"])
        ax.set_facecolor(PALETTE["background"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["ink"])

    fig.tight_layout()
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_brushstroke_map(
    report: BrushstrokeReport,
    original_image: Image,
    path: str | Path,
    *,
    dpi: int = 300,
) -> None:
    """Overlay brushstroke cluster assignments onto the original artwork.

    Parameters
    ----------
    report:
        Brushstroke analysis report.
    original_image:
        Original artwork image.
    path:
        Output file path.
    dpi:
        Output resolution.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=PALETTE["background"])
    ax.imshow(np.array(original_image))

    cluster_colors = ["#D4899A", "#9DC0D8", "#d4af37", "#7c2d12"]

    if report.cluster_labels is not None:
        for desc, label in zip(report.descriptors, report.cluster_labels, strict=False):
            x, y, w, h = desc.bbox
            color = cluster_colors[int(label) % len(cluster_colors)]
            rect = mpatches.Rectangle(
                (x, y), w, h,
                linewidth=1.5,
                edgecolor=color,
                facecolor=color,
                alpha=0.25,
            )
            ax.add_patch(rect)

    ax.axis("off")
    ax.set_title("Brushstroke Cluster Map", fontsize=14, color=PALETTE["ink"])
    fig.tight_layout()
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_analysis_summary(
    result: AnalysisResult,
    path: str | Path,
    *,
    dpi: int = 300,
) -> None:
    """Render a multi-panel summary figure for a complete analysis.

    Combines the original artwork, style distribution, brushstroke
    cluster map, and attribution ranking into a single composite
    suitable for presentations or reports.

    Parameters
    ----------
    result:
        Complete ArtSleuth analysis result.
    path:
        Output file path.
    dpi:
        Output resolution.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    fig = plt.figure(figsize=(20, 10), facecolor=PALETTE["background"])

    # Left panel — original artwork
    ax_img = fig.add_subplot(1, 2, 1)
    original = PILImage.open(result.image_path).convert("RGB")
    ax_img.imshow(np.array(original))
    ax_img.set_title("Original", fontsize=14, color=PALETTE["ink"])
    ax_img.axis("off")

    # Right panel — textual summary
    ax_text = fig.add_subplot(1, 2, 2)
    ax_text.axis("off")
    ax_text.set_facecolor(PALETTE["background"])

    summary_text = result.summary()
    ax_text.text(
        0.05,
        0.95,
        summary_text,
        transform=ax_text.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="serif",
        color=PALETTE["ink"],
        linespacing=1.8,
    )

    fig.tight_layout()
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
