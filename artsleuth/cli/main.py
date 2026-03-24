"""
Command-line interface for ArtSleuth.

For when you just want to point at a painting and get answers without
opening a notebook.  Rich terminal output for readable, colour-formatted
results.
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
    """ArtSleuth — Computational Art Analysis Framework."""


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


# British spelling alias
@cli.command()
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option("--reference-artist", "-r", default=None, help="Artist name for forgery screening.")
@click.option("--output", "-o", default=None, type=click.Path(path_type=Path), help="Save summary figure.")
@click.option("--device", "-d", default=None, help="PyTorch device.")
def analyse(image: Path, reference_artist: str | None, output: Path | None, device: str | None) -> None:
    """Run the full analysis pipeline on an artwork (alias for analyze)."""
    analyze.callback(image, reference_artist, output, device)  # type: ignore[union-attr]


# --- Style ------------------------------------------------------------------


@cli.command()
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option("--top-k", "-k", default=5, help="Number of predictions per axis.")
@click.option("--device", "-d", default=None, help="PyTorch device.")
def style(image: Path, top_k: int, device: str | None) -> None:
    """Classify artwork by period, school, and genre."""
    from PIL import Image

    from artsleuth.config import AnalysisConfig
    from artsleuth.core.style import StyleClassifier

    with console.status("[bold blue]Classifying style…", spinner="aesthetic"):
        config = AnalysisConfig(device=device)
        classifier = StyleClassifier(config)
        img = Image.open(str(image)).convert("RGB")
        report = classifier.classify(img, top_k=top_k)

    axes = [("Period", report.period), ("School", report.school), ("Genre", report.technique)]
    for axis_name, pred in axes:
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


# --- Workshop ---------------------------------------------------------------


@cli.command()
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option("--max-hands", default=6, help="Maximum number of hands to infer.")
@click.option("--device", "-d", default=None, help="PyTorch device.")
def workshop(image: Path, max_hands: int, device: str | None) -> None:
    """Decompose a painting into distinct workshop hands."""
    import numpy as np
    from PIL import Image

    from artsleuth.config import AnalysisConfig
    from artsleuth.core.brushstroke import BrushstrokeAnalyzer
    from artsleuth.core.workshop import WorkshopDecomposition

    with console.status("[bold]Decomposing workshop hands…", spinner="aesthetic"):
        config = AnalysisConfig(device=device)
        analyzer = BrushstrokeAnalyzer(config)
        img = Image.open(str(image)).convert("RGB")
        report = analyzer.analyze(img)

        embeddings = np.stack([d.embedding for d in report.descriptors])
        bboxes = [d.bbox for d in report.descriptors]
        coherences = np.array([d.coherence for d in report.descriptors])
        energies = np.array([d.energy for d in report.descriptors])

        decomposer = WorkshopDecomposition(max_hands=max_hands)
        ws = decomposer.decompose(
            embeddings, bboxes, img.size,
            coherences=coherences, energies=energies,
        )

    table = Table(title="Workshop Decomposition", border_style="dim")
    table.add_column("Hand", style="bold")
    table.add_column("Patches", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Coherence", justify="right")
    table.add_column("Energy", justify="right")
    table.add_column("Confidence", justify="right")

    for a in ws.assignments:
        table.add_row(
            a.label,
            str(a.patch_count),
            f"{a.spatial_extent:.1%}",
            f"{a.mean_coherence:.3f}",
            f"{a.mean_energy:.3f}",
            f"{a.confidence:.1%}",
        )

    console.print(table)
    if ws.is_workshop:
        console.print(
            f"\n[bold]Workshop production detected:[/bold] "
            f"{ws.num_hands} distinct hands identified."
        )
    else:
        console.print("\n[dim]Single-hand execution (no workshop detected).[/dim]")


# --- Robustness -------------------------------------------------------------


@cli.command()
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--reference-artist", "-r", required=True,
    help="Artist name for forgery screening.",
)
@click.option("--device", "-d", default=None, help="PyTorch device.")
def robustness(image: Path, reference_artist: str, device: str | None) -> None:
    """Test forgery detection robustness against adversarial techniques."""
    from PIL import Image

    from artsleuth.config import AnalysisConfig
    from artsleuth.core.adversarial import ForgerySimulator, RobustnessEvaluator
    from artsleuth.core.forgery import ForgeryDetector

    with console.status("[bold red]Running adversarial robustness tests…", spinner="aesthetic"):
        config = AnalysisConfig(device=device)
        detector = ForgeryDetector(config)
        simulator = ForgerySimulator()
        evaluator = RobustnessEvaluator(detector=detector, simulator=simulator)

        img = Image.open(str(image)).convert("RGB")
        report = evaluator.evaluate(img, reference_artist)

    table = Table(title="Adversarial Robustness", border_style="dim")
    table.add_column("Technique", style="bold")
    table.add_column("Detected", justify="center")
    table.add_column("Score Delta", justify="right")

    for ar in report.technique_results:
        detected_str = "[green]Yes[/green]" if ar.detected else "[red]No[/red]"
        table.add_row(
            ar.technique.name,
            detected_str,
            f"{ar.score_delta:+.3f}",
        )

    console.print(table)
    console.print(
        f"\n[bold]Overall detection rate:[/bold] {report.overall_detection_rate:.0%}"
    )


# --- Benchmark --------------------------------------------------------------


@cli.command()
@click.option(
    "--backbone", "-b", default=None, multiple=True,
    help="Backbone(s) to benchmark (dinov2, clip, fusion). Omit for all.",
)
@click.option("--max-samples", default=None, type=int, help="Cap samples per split.")
@click.option("--device", "-d", default=None, help="PyTorch device.")
@click.option(
    "--output-dir", "-o", default="benchmark_results",
    type=click.Path(path_type=Path),
)
def benchmark(
    backbone: tuple[str, ...],
    max_samples: int | None,
    device: str | None,
    output_dir: Path,
) -> None:
    """Run WikiArt benchmarks and produce comparison tables."""
    from artsleuth.benchmarks.evaluate import run_all_benchmarks

    backbones = list(backbone) if backbone else None

    with console.status("[bold]Running benchmarks…", spinner="aesthetic"):
        table = run_all_benchmarks(
            backbones=backbones,
            device=device or "cpu",
            max_samples=max_samples,
            output_dir=output_dir,
        )

    console.print(table.to_markdown())
    console.print(f"\n[dim]Full results saved to {output_dir}/[/dim]")


# --- Demo (Web UI) ----------------------------------------------------------


@cli.command()
@click.option("--port", "-p", default=7860, help="Port for the Gradio server.")
@click.option("--share", is_flag=True, help="Create a public Gradio link.")
def demo(port: int, share: bool) -> None:
    """Launch the ArtSleuth web demo."""
    try:
        from web.app import create_app
    except ImportError:
        console.print(
            "[red]Web UI dependencies not installed. "
            "Run: pip install artsleuth[web][/red]"
        )
        return

    app = create_app()
    app.launch(server_port=port, share=share)


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
