"""
Command-line interface for ArtSleuth.

Provides quick access to all analysis capabilities from the terminal
with rich, colour-formatted output.

Usage
-----
::

    artsleuth analyze painting.jpg
    artsleuth style painting.jpg
    artsleuth compare painting_a.jpg painting_b.jpg
    artsleuth server
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


# --- Main Group -------------------------------------------------------------


@click.group()
@click.version_option(package_name="artsleuth")
def cli() -> None:
    """ArtSleuth — AI Art Forensics & Analysis Framework."""


# --- Analyze ----------------------------------------------------------------


@cli.command()
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--reference-artist",
    "-r",
    default=None,
    help="Artist name for forgery screening.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Save analysis summary figure to this path.",
)
@click.option("--device", "-d", default=None, help="PyTorch device (cuda, mps, cpu).")
def analyze(
    image: Path,
    reference_artist: str | None,
    output: Path | None,
    device: str | None,
) -> None:
    """Run the full analysis pipeline on an artwork."""
    from artsleuth.config import AnalysisConfig
    from artsleuth.core.pipeline import run_pipeline

    with console.status("[bold rose]Analysing artwork…", spinner="aesthetic"):
        config = AnalysisConfig(device=device)
        result = run_pipeline(
            str(image),
            config=config,
            reference_artist=reference_artist,
        )

    _render_result(result)

    if output:
        from artsleuth.utils.visualization import render_analysis_summary

        render_analysis_summary(result, output)
        console.print(f"\n[dim]Summary figure saved to {output}[/dim]")


# --- Style ------------------------------------------------------------------


@cli.command()
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option("--top-k", "-k", default=5, help="Number of predictions per axis.")
@click.option("--device", "-d", default=None, help="PyTorch device.")
def style(image: Path, top_k: int, device: str | None) -> None:
    """Classify artwork by period, school, and technique."""
    from PIL import Image

    from artsleuth.config import AnalysisConfig
    from artsleuth.core.style import StyleClassifier

    with console.status("[bold blue]Classifying style…", spinner="aesthetic"):
        config = AnalysisConfig(device=device)
        classifier = StyleClassifier(config)
        img = Image.open(str(image)).convert("RGB")
        report = classifier.classify(img, top_k=top_k)

    for axis_name, pred in [("Period", report.period), ("School", report.school), ("Technique", report.technique)]:
        table = Table(title=axis_name, title_style="bold", border_style="dim")
        table.add_column("Label", style="bold")
        table.add_column("Confidence", justify="right")

        for label, conf in pred.top_k:
            bar = "█" * int(conf * 30) + "░" * (30 - int(conf * 30))
            table.add_row(label, f"{bar} {conf:.1%}")

        console.print(table)
        console.print()


# --- Compare ----------------------------------------------------------------


@cli.command()
@click.argument("image_a", type=click.Path(exists=True, path_type=Path))
@click.argument("image_b", type=click.Path(exists=True, path_type=Path))
@click.option("--device", "-d", default=None, help="PyTorch device.")
def compare(image_a: Path, image_b: Path, device: str | None) -> None:
    """Compare two artworks for stylistic similarity."""
    import numpy as np
    from PIL import Image

    from artsleuth.config import AnalysisConfig
    from artsleuth.core.style import StyleClassifier

    with console.status("[bold gold1]Comparing artworks…", spinner="aesthetic"):
        config = AnalysisConfig(device=device)
        classifier = StyleClassifier(config)

        img_a = Image.open(str(image_a)).convert("RGB")
        img_b = Image.open(str(image_b)).convert("RGB")

        report_a = classifier.classify(img_a)
        report_b = classifier.classify(img_b)

        similarity = float(
            np.dot(report_a.embedding, report_b.embedding)
            / (np.linalg.norm(report_a.embedding) * np.linalg.norm(report_b.embedding) + 1e-12)
        )

    interpretation = (
        "Very likely same artist or workshop" if similarity > 0.85
        else "Probable stylistic relationship" if similarity > 0.65
        else "Some shared characteristics" if similarity > 0.45
        else "Distinct stylistic profiles"
    )

    panel_text = Text()
    panel_text.append(f"Cosine Similarity: {similarity:.3f}\n", style="bold")
    panel_text.append(f"Interpretation: {interpretation}\n\n")
    panel_text.append(f"  A: {report_a.period.label}, {report_a.school.label}\n", style="dim")
    panel_text.append(f"  B: {report_b.period.label}, {report_b.school.label}", style="dim")

    console.print(Panel(panel_text, title="Comparison", border_style="gold1"))


# --- Server -----------------------------------------------------------------


@cli.command()
@click.option("--transport", "-t", default="stdio", type=click.Choice(["stdio", "sse"]))
def server(transport: str) -> None:
    """Launch the ArtSleuth MCP server."""
    from artsleuth.mcp.server import create_server

    console.print("[bold]Starting ArtSleuth MCP server…[/bold]")
    mcp_server = create_server()

    if transport == "stdio":
        import asyncio

        from mcp.server.stdio import stdio_server

        async def _run() -> None:
            async with stdio_server() as (read_stream, write_stream):
                await mcp_server.run(read_stream, write_stream)

        asyncio.run(_run())
    else:
        console.print("[red]SSE transport not yet implemented.[/red]")


# --- Helpers ----------------------------------------------------------------


def _render_result(result: "object") -> None:
    """Pretty-print an AnalysisResult to the console."""
    from artsleuth.core.pipeline import AnalysisResult

    if not isinstance(result, AnalysisResult):
        return

    console.print()
    console.print(
        Panel(
            result.summary(),
            title="[bold]ArtSleuth Analysis[/bold]",
            border_style="bright_cyan",
            padding=(1, 2),
        )
    )
