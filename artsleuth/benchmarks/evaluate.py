"""
Unified benchmark runner.

Orchestrates WikiArt evaluation across all backbone configurations
and produces comparison tables for the README and methodology docs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --- Data Structures --------------------------------------------------------


@dataclass
class ComparisonRow:
    """A single row in the benchmark comparison table."""

    backbone: str
    style_acc: float = 0.0
    style_f1: float = 0.0
    artist_acc: float = 0.0
    artist_top5: float = 0.0
    genre_acc: float = 0.0


@dataclass
class ComparisonTable:
    """Full benchmark comparison across backbones."""

    rows: list[ComparisonRow] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the comparison as a Markdown table."""
        lines = [
            "| Backbone | Style Acc | Style F1 | Artist Acc | Artist Top-5 | Genre Acc |",
            "|:---------|:---------:|:--------:|:----------:|:------------:|:---------:|",
        ]
        for r in self.rows:
            lines.append(
                f"| **{r.backbone}** "
                f"| {r.style_acc:.1%} "
                f"| {r.style_f1:.3f} "
                f"| {r.artist_acc:.1%} "
                f"| {r.artist_top5:.1%} "
                f"| {r.genre_acc:.1%} |"
            )
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Render the comparison as a LaTeX tabular."""
        lines = [
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Backbone & Style Acc & Style F1 & Artist Acc & Artist Top-5 & Genre Acc \\",
            r"\midrule",
        ]
        for r in self.rows:
            lines.append(
                f"\\textbf{{{r.backbone}}} & "
                f"{r.style_acc:.1%} & {r.style_f1:.3f} & "
                f"{r.artist_acc:.1%} & {r.artist_top5:.1%} & "
                f"{r.genre_acc:.1%} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return "\n".join(lines)


# --- Runner -----------------------------------------------------------------


def run_all_benchmarks(
    *,
    backbones: list[str] | None = None,
    device: str = "cpu",
    max_samples: int | None = None,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> ComparisonTable:
    """Run WikiArt benchmarks across all backbone configurations.

    Parameters
    ----------
    backbones:
        List of backbone names to evaluate.  Defaults to
        ``["dinov2", "clip", "fusion"]``.
    device:
        PyTorch device.
    max_samples:
        Cap per split for faster iteration.
    output_dir:
        Directory for JSON and table outputs.
    cache_dir:
        Weight and dataset cache directory.

    Returns
    -------
    ComparisonTable
        Aggregated benchmark metrics.
    """
    from artsleuth.benchmarks.wikiart import run_wikiart_benchmark

    if backbones is None:
        backbones = ["dinov2", "clip", "fusion"]

    if output_dir is None:
        output_dir = Path("benchmark_results")

    output_dir.mkdir(parents=True, exist_ok=True)

    table = ComparisonTable()

    for bb in backbones:
        logger.info("=" * 60)
        logger.info("Benchmarking: %s", bb)
        logger.info("=" * 60)

        result = run_wikiart_benchmark(
            backbone=bb,
            device=device,
            max_samples=max_samples,
            output_dir=output_dir,
            cache_dir=cache_dir,
        )

        row = ComparisonRow(backbone=bb)
        if result.style_metrics:
            row.style_acc = result.style_metrics.accuracy
            row.style_f1 = result.style_metrics.macro_f1
        if result.artist_metrics:
            row.artist_acc = result.artist_metrics.accuracy
            row.artist_top5 = result.artist_metrics.top5_accuracy
        if result.genre_metrics:
            row.genre_acc = result.genre_metrics.accuracy

        table.rows.append(row)

    # Save comparison tables
    md_path = output_dir / "comparison.md"
    md_path.write_text(table.to_markdown(), encoding="utf-8")
    logger.info("Markdown table saved to %s", md_path)

    latex_path = output_dir / "comparison.tex"
    latex_path.write_text(table.to_latex(), encoding="utf-8")
    logger.info("LaTeX table saved to %s", latex_path)

    # Save raw JSON in the same schema as benchmark_results.json
    frozen_dict: dict[str, Any] = {}
    for r in table.rows:
        frozen_dict[r.backbone] = {
            "style": {
                "accuracy": r.style_acc,
                "macro_f1": r.style_f1,
            },
            "artist": {
                "accuracy": r.artist_acc,
                "top5_accuracy": r.artist_top5,
            },
            "genre": {
                "accuracy": r.genre_acc,
            },
        }
    results = {
        "frozen": frozen_dict,
        "_note_tuned": (
            "Fine-tuned and e2e results require a separate training run "
            "not included in this evaluation script."
        ),
    }
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return table
