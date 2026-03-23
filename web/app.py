"""
ArtSleuth web demo.

Interactive art analysis powered by vision transformers. Upload a
painting and get brushstroke analysis, style classification, artist
attribution, forgery screening, and workshop decomposition — all
with publication-quality visualizations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np


def _error_html(message: str) -> str:
    """Wrap an error message in styled HTML."""
    return (
        '<div style="color:#b91c1c;background:#fef2f2;'
        'border:1px solid #fecaca;border-radius:8px;'
        f'padding:1rem;margin:0.5rem 0">'
        f"<strong>Error:</strong> {message}</div>"
    )


def _info_html(message: str) -> str:
    """Wrap an informational message in styled HTML."""
    return (
        '<div style="color:#1e40af;background:#eff6ff;'
        'border:1px solid #bfdbfe;border-radius:8px;'
        f'padding:1rem;margin:0.5rem 0">{message}</div>'
    )


def _save_pil_to_temp(image) -> str:
    """Persist a PIL image to a temporary PNG and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".png", delete=False,
    )
    image.save(tmp, format="PNG")
    tmp.close()
    return tmp.name


# ── UI builder ──────────────────────────────────────────────────────


def create_app() -> gr.Blocks:
    """Build the ArtSleuth Gradio application.

    Returns
    -------
    gr.Blocks
        Fully wired Gradio Blocks application ready to ``.launch()``.
    """
    from web.theme import artsleuth_theme, CUSTOM_CSS, HEADER_HTML, FOOTER_HTML

    # ── handler: Analyze ────────────────────────────────────────────

    def _handle_analyze(image, reference_artist: str):
        """Run the full analysis pipeline on a single artwork."""
        try:
            from artsleuth.config import AnalysisConfig
            from artsleuth.core.pipeline import run_pipeline
            from web.components import (
                format_style_report,
                format_attribution_report,
                format_forgery_gauge,
            )

            if image is None:
                empty = _info_html("Upload an artwork to begin.")
                return empty, empty, empty, None

            path = _save_pil_to_temp(image)
            ref = reference_artist.strip() or None

            config = AnalysisConfig()
            result = run_pipeline(
                path, config=config, reference_artist=ref,
            )

            style_html = format_style_report(result.style)
            attribution_html = format_attribution_report(
                result.attribution,
            )

            forgery_html = ""
            if result.forgery is not None:
                forgery_html = format_forgery_gauge(result.forgery)
            else:
                forgery_html = _info_html(
                    "Provide a reference artist to enable "
                    "forgery screening."
                )

            heatmap_image = None
            try:
                explanation = result.explain(target="attribution")
                if explanation.composite is not None:
                    heatmap_image = explanation.composite
            except Exception:
                forgery_html += _info_html(
                    "Saliency heatmap could not be generated for this image."
                )

            return (
                style_html,
                attribution_html,
                forgery_html,
                heatmap_image,
            )
        except Exception as exc:
            err = _error_html(str(exc))
            return err, err, err, None

    # ── handler: Compare ────────────────────────────────────────────

    def _handle_compare(image_a, image_b):
        """Classify both images and compute cosine similarity."""
        try:
            from artsleuth.config import AnalysisConfig
            from artsleuth.core.style import StyleClassifier

            if image_a is None or image_b is None:
                msg = _info_html(
                    "Upload two artworks to compare."
                )
                return msg, msg

            config = AnalysisConfig()
            classifier = StyleClassifier(config)

            report_a = classifier.classify(image_a)
            report_b = classifier.classify(image_b)

            emb_a = report_a.embedding
            emb_b = report_b.embedding
            dot = float(np.dot(emb_a, emb_b))
            norm = (
                float(np.linalg.norm(emb_a))
                * float(np.linalg.norm(emb_b))
                + 1e-12
            )
            similarity = dot / norm

            if similarity > 0.85:
                interp = (
                    "Very high embedding similarity — the works share "
                    "strong stylistic features. This does not confirm "
                    "shared authorship on its own."
                )
            elif similarity > 0.65:
                interp = (
                    "Moderate similarity — some shared stylistic "
                    "traits, but distinct in other respects."
                )
            elif similarity > 0.40:
                interp = (
                    "Low similarity — different stylistic profiles "
                    "with some overlapping features."
                )
            else:
                interp = "Minimal similarity — markedly different stylistic profiles."

            sim_html = (
                '<div style="text-align:center;padding:1.5rem">'
                '<div style="font-size:3rem;font-weight:700;'
                f'color:#1A2E48">{similarity:.1%}</div>'
                '<div style="font-size:1.1rem;color:#64748b;'
                f'margin-top:0.5rem">{interp}</div></div>'
            )

            def _axis_row(label, pred_a, pred_b):
                return (
                    f"<tr><td><strong>{label}</strong></td>"
                    f"<td>{pred_a.label} ({pred_a.confidence:.0%})</td>"
                    f"<td>{pred_b.label} ({pred_b.confidence:.0%})</td>"
                    "</tr>"
                )

            comparison_html = (
                '<table style="width:100%;border-collapse:collapse;'
                'margin-top:1rem">'
                '<thead><tr style="border-bottom:2px solid #e2e8f0">'
                "<th>Axis</th><th>Artwork A</th><th>Artwork B</th>"
                "</tr></thead><tbody>"
                + _axis_row(
                    "Period", report_a.period, report_b.period,
                )
                + _axis_row(
                    "School", report_a.school, report_b.school,
                )
                + _axis_row(
                    "Genre",
                    report_a.technique,
                    report_b.technique,
                )
                + "</tbody></table>"
            )

            return sim_html, comparison_html
        except Exception as exc:
            err = _error_html(str(exc))
            return err, err

    # ── handler: Workshop ───────────────────────────────────────────

    def _handle_workshop(image, max_hands: int):
        """Run brushstroke analysis + workshop decomposition."""
        try:
            from artsleuth.config import AnalysisConfig
            from artsleuth.core.brushstroke import BrushstrokeAnalyzer
            from artsleuth.core.workshop import WorkshopDecomposition
            from web.components import format_workshop_report

            if image is None:
                msg = _info_html("Upload an artwork to decompose.")
                return msg, None

            config = AnalysisConfig(
                workshop_max_hands=int(max_hands),
            )
            analyzer = BrushstrokeAnalyzer(config)
            brushstroke_report = analyzer.analyze(image)

            if not brushstroke_report.descriptors:
                return (
                    _info_html(
                        "No brushstroke patches extracted. "
                        "Try a higher-resolution image."
                    ),
                    None,
                )

            embeddings = np.stack(
                [d.embedding for d in brushstroke_report.descriptors]
            )
            bboxes = [
                d.bbox for d in brushstroke_report.descriptors
            ]
            coherences = np.array(
                [d.coherence for d in brushstroke_report.descriptors]
            )
            energies = np.array(
                [d.energy for d in brushstroke_report.descriptors]
            )

            decomposer = WorkshopDecomposition(
                max_hands=int(max_hands),
            )
            workshop_report = decomposer.decompose(
                embeddings,
                bboxes,
                image.size,
                coherences=coherences,
                energies=energies,
            )

            report_html = format_workshop_report(workshop_report)
            hand_map_image = None
            if workshop_report.hand_map is not None:
                from PIL import Image as PILImage

                unique_ids = np.unique(workshop_report.hand_map)
                valid_ids = unique_ids[unique_ids >= 0]
                n_hands = len(valid_ids)
                palette = _hand_map_palette(n_hands)

                h, w = workshop_report.hand_map.shape
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                for hid in valid_ids:
                    mask = workshop_report.hand_map == hid
                    canvas[mask] = palette[int(hid) % len(palette)]

                orig = np.array(image.convert("RGB"))
                if orig.shape[:2] == (h, w):
                    blended = (
                        0.6 * orig.astype(np.float32)
                        + 0.4 * canvas.astype(np.float32)
                    )
                    hand_map_image = blended.astype(np.uint8)
                else:
                    hand_map_image = canvas

            return report_html, hand_map_image
        except Exception as exc:
            return _error_html(str(exc)), None

    # ── handler: Timeline ───────────────────────────────────────────

    def _handle_timeline(image, artist_name: str):
        """Run style classification + temporal estimation."""
        try:
            from artsleuth.config import AnalysisConfig
            from artsleuth.core.style import StyleClassifier
            from artsleuth.core.temporal import TemporalRegistry

            if image is None:
                return _info_html("Upload an artwork to estimate.")
            if not artist_name.strip():
                return _info_html(
                    "Enter an artist name for temporal analysis."
                )

            config = AnalysisConfig()
            classifier = StyleClassifier(config)
            style_report = classifier.classify(image)

            registry = TemporalRegistry()
            prediction = registry.predict(
                artist_name.strip(), style_report.embedding,
            )

            if prediction is None:
                return _info_html(
                    f"No temporal model available for "
                    f"'{artist_name.strip()}'. The registry "
                    f"requires dated reference works."
                )

            lo, hi = prediction.confidence_band
            return (
                '<div style="text-align:center;padding:1.5rem">'
                '<div style="font-size:2.5rem;font-weight:700;'
                f'color:#1A2E48">c.\u2009{prediction.estimated_year:.0f}'
                "</div>"
                '<div style="color:#64748b;margin-top:0.5rem">'
                f"95% band: {lo:.0f}\u2013{hi:.0f}</div>"
                '<div style="margin-top:1rem">'
                f"Temporal plausibility: "
                f"<strong>{prediction.temporal_score:.0%}</strong>"
                f" &nbsp;|&nbsp; Drift rate: "
                f"<strong>{prediction.drift_rate:.3f}</strong> / decade"
                "</div></div>"
            )
        except Exception as exc:
            return _error_html(str(exc))

    # ── layout ──────────────────────────────────────────────────────

    with gr.Blocks(
        theme=artsleuth_theme(),
        css=CUSTOM_CSS,
    ) as app:
        gr.HTML(HEADER_HTML)

        with gr.Tabs():
            # ── Tab 1: Analyze ──────────────────────────────────
            with gr.Tab("Analyze"):
                with gr.Row():
                    with gr.Column(scale=1):
                        analyze_image = gr.Image(
                            type="pil", label="Upload Artwork",
                        )
                        analyze_ref = gr.Textbox(
                            label="Reference Artist (optional)",
                            placeholder="e.g. Rembrandt",
                        )
                        analyze_btn = gr.Button(
                            "Analyze", variant="primary",
                        )
                    with gr.Column(scale=2):
                        analyze_style = gr.HTML(
                            label="Style Report",
                        )
                        analyze_attr = gr.HTML(
                            label="Attribution Report",
                        )
                        analyze_forgery = gr.HTML(
                            label="Forgery Screening",
                        )
                        analyze_heatmap = gr.Image(
                            label="Saliency Heatmap",
                        )

                analyze_btn.click(
                    fn=_handle_analyze,
                    inputs=[analyze_image, analyze_ref],
                    outputs=[
                        analyze_style,
                        analyze_attr,
                        analyze_forgery,
                        analyze_heatmap,
                    ],
                )

            # ── Tab 2: Compare ──────────────────────────────────
            with gr.Tab("Compare"):
                with gr.Row():
                    compare_a = gr.Image(
                        type="pil", label="Artwork A",
                    )
                    compare_b = gr.Image(
                        type="pil", label="Artwork B",
                    )
                compare_btn = gr.Button(
                    "Compare", variant="primary",
                )
                compare_sim = gr.HTML(label="Similarity")
                compare_axes = gr.HTML(label="Style Comparison")

                compare_btn.click(
                    fn=_handle_compare,
                    inputs=[compare_a, compare_b],
                    outputs=[compare_sim, compare_axes],
                )

            # ── Tab 3: Workshop ─────────────────────────────────
            with gr.Tab("Workshop"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ws_image = gr.Image(
                            type="pil", label="Upload Artwork",
                        )
                        ws_hands = gr.Slider(
                            minimum=2,
                            maximum=8,
                            value=6,
                            step=1,
                            label="Max Hands",
                        )
                        ws_btn = gr.Button(
                            "Decompose", variant="primary",
                        )
                    with gr.Column(scale=2):
                        ws_report = gr.HTML(
                            label="Workshop Report",
                        )
                        ws_map = gr.Image(
                            label="Hand Map Overlay",
                        )

                ws_btn.click(
                    fn=_handle_workshop,
                    inputs=[ws_image, ws_hands],
                    outputs=[ws_report, ws_map],
                )

            # ── Tab 4: Timeline ─────────────────────────────────
            with gr.Tab("Timeline"):
                with gr.Row():
                    with gr.Column(scale=1):
                        tl_image = gr.Image(
                            type="pil", label="Upload Artwork",
                        )
                        tl_artist = gr.Textbox(
                            label="Artist Name",
                            placeholder="e.g. Artemisia Gentileschi",
                        )
                        tl_btn = gr.Button(
                            "Estimate Date", variant="primary",
                        )
                    with gr.Column(scale=2):
                        tl_result = gr.HTML(
                            label="Temporal Prediction",
                        )

                tl_btn.click(
                    fn=_handle_timeline,
                    inputs=[tl_image, tl_artist],
                    outputs=[tl_result],
                )

            # ── Tab 5: Benchmark ────────────────────────────────
            with gr.Tab("Benchmark"):
                gr.HTML(_build_benchmark_table())
                gr.Markdown(_benchmark_methodology())

        gr.HTML(FOOTER_HTML)

    return app


# ── helpers ─────────────────────────────────────────────────────────


def _hand_map_palette(n: int) -> list[list[int]]:
    """Return *n* visually distinct RGB colours for hand-map overlays."""
    base = [
        [212, 137, 154],  # rose
        [157, 192, 216],  # blue
        [212, 175, 55],   # gold
        [124, 45, 18],    # umber
        [74, 144, 120],   # sage
        [180, 120, 200],  # lavender
        [220, 170, 120],  # sand
        [100, 100, 180],  # slate
    ]
    return base[:max(n, 1)]


def _build_benchmark_table() -> str:
    """Return an HTML table with WikiArt benchmark numbers."""
    row = (
        '<tr style="border-bottom:{border}">'
        '<td style="padding:0.5rem 1rem;text-align:left;{extra}">{name}</td>'
        '<td style="padding:0.5rem 0.7rem;{extra}">{style}</td>'
        '<td style="padding:0.5rem 0.7rem;{extra}">{f1}</td>'
        '<td style="padding:0.5rem 0.7rem;{extra}">{artist}</td>'
        '<td style="padding:0.5rem 0.7rem;{extra}">{top5}</td>'
        '<td style="padding:0.5rem 0.7rem;{extra}">{genre}</td></tr>'
    )
    thin = "1px solid #d4c5b9"
    thick = "2px solid #1A2E48"
    rows = [
        row.format(border=thin, extra="", name="DINOv2 · ViT-B/14",
                   style="57.5%", f1="0.553", artist="64.7%",
                   top5="90.9%", genre="71.0%"),
        row.format(border=thin, extra="", name="CLIP · ViT-L/14",
                   style="67.1%", f1="0.656", artist="74.6%",
                   top5="95.9%", genre="75.0%"),
        row.format(border=thin, extra="", name="Fusion · frozen",
                   style="65.0%", f1="0.633", artist="71.0%",
                   top5="94.2%", genre="74.2%"),
        row.format(border=thin, extra="", name="Fusion · fine-tuned",
                   style="71.6%", f1="0.703", artist="77.8%",
                   top5="96.2%", genre="75.1%"),
        row.format(border=thick, extra="font-weight:600",
                   name="Fusion · e2e", style="72.7%", f1="—",
                   artist="79.0%", top5="96.9%", genre="76.6%"),
    ]
    return (
        '<div style="text-align:center;padding:2rem 1rem;'
        'background:#F5F0EB;border-radius:8px;margin:1rem 0">'
        '<h3 style="color:#1A2E48;margin:0 0 0.5rem">'
        'WikiArt Benchmark (81 444 images)</h3>'
        '<table style="margin:1rem auto;border-collapse:collapse;'
        'font-size:0.9rem;color:#1A2E48">'
        '<tr style="border-bottom:2px solid #1A2E48">'
        '<th style="padding:0.5rem 1rem;text-align:left">Backbone</th>'
        '<th style="padding:0.5rem 0.7rem">Style</th>'
        '<th style="padding:0.5rem 0.7rem">Style F1</th>'
        '<th style="padding:0.5rem 0.7rem">Artist</th>'
        '<th style="padding:0.5rem 0.7rem">Artist Top-5</th>'
        '<th style="padding:0.5rem 0.7rem">Genre</th></tr>'
        + "".join(rows)
        + '</table>'
        '<p style="color:#988b7e;font-size:0.78rem;margin-top:0.5rem">'
        'Top four rows: logistic-regression linear probes. Bottom row: '
        'end-to-end classification heads. Fine-tuning uses SupCon + CE loss, '
        'partial backbone unfreezing (3 blocks), cosine annealing, 5 epochs.'
        '</p>'
        '<p style="color:#988b7e;font-size:0.72rem;margin-top:0.3rem;font-style:italic">'
        'These numbers are pre-computed from the published benchmark run. '
        'They do not update dynamically.</p></div>'
    )


def _benchmark_methodology() -> str:
    """Return Markdown text explaining the benchmark methodology."""
    return (
        "### Methodology\n\n"
        "All models are evaluated on the full **WikiArt** dataset "
        "(81 444 images, 80/20 split, 27 styles, 129 artists, 11 genres). "
        "Style and genre accuracy use single-label top-1 matching; "
        "artist accuracy reports both top-1 and top-5.\n\n"
        "**ArtSleuth Fusion** combines DINOv2 ViT-B/14 texture features "
        "with CLIP ViT-L/14 semantic embeddings via a learnable "
        "cross-attention fusion head.\n\n"
        "The **fine-tuned** variant partially unfreezes the last 3 "
        "transformer blocks of each backbone and trains with a multi-task "
        "objective (cross-entropy + supervised contrastive loss, "
        "weight 0.2). The **e2e** row reports the jointly trained "
        "classification heads directly, without a separate linear probe.\n\n"
        "Training uses AdamW with cosine annealing (backbone lr=1e-5, "
        "head lr=5e-4), mixed precision, gradient accumulation (effective "
        "batch 64), for 5 epochs on a Tesla P100.\n\n"
        "*All numbers are macro-averaged across classes to avoid "
        "inflating scores on over-represented styles.*"
    )


# ── entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    app.launch()
