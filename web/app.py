"""
ArtSleuth web demo.

Interactive art analysis powered by vision transformers. Upload a
painting and get brushstroke analysis, style classification, artist
attribution, forgery screening, and workshop decomposition — all
with publication-quality visualisations.
"""

from __future__ import annotations

import html as html_module
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np


def _error_html(message: str) -> str:
    """Wrap an error message in a callout."""
    safe = html_module.escape(str(message))
    return (
        '<div class="as-callout as-callout--error">'
        f"<strong>Error:</strong> {safe}</div>"
    )


def _info_html(message: str) -> str:
    """Wrap an informational message in a callout."""
    safe = html_module.escape(str(message))
    return f'<div class="as-callout as-callout--info">{safe}</div>'


def _save_pil_to_temp(image) -> str:
    """Persist a PIL image to a temporary PNG and return the path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp, format="PNG")
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

    # ── handler: Analyse ────────────────────────────────────────────

    def _handle_analyse(image, reference_artist: str):
        """Run the full analysis pipeline on a single artwork."""
        try:
            from artsleuth.config import AnalysisConfig
            from artsleuth.core.pipeline import run_pipeline
            from artsleuth.core.style import StyleClassifier
            from web.components import (
                format_style_report,
                format_attribution_report,
                format_forgery_gauge,
            )

            if image is None:
                empty = _info_html("Upload an artwork to begin.")
                return empty, empty, empty, empty, None

            path = _save_pil_to_temp(image)
            ref = reference_artist.strip() or None

            config = AnalysisConfig()
            result = run_pipeline(
                path, config=config, reference_artist=ref,
            )

            style_html = format_style_report(result.style)

            try:
                classifier = StyleClassifier(config)
                artist_top = classifier.estimate_artist(
                    result.style.embedding, top_k=5,
                )
            except Exception:
                artist_top = []
            artist_html = _format_artist_estimation(artist_top)

            attribution_html = format_attribution_report(
                result.attribution,
            )

            forgery_html = ""
            if result.forgery is not None:
                forgery_html = format_forgery_gauge(result.forgery)
            else:
                forgery_html = _info_html(
                    "Provide a reference artist to enable "
                    "forgery screening.",
                )

            heatmap_image = None
            try:
                explanation = result.explain(target="attribution")
                if explanation.composite is not None:
                    heatmap_image = explanation.composite
            except Exception:
                pass

            return (
                style_html,
                artist_html,
                attribution_html,
                forgery_html,
                heatmap_image,
            )
        except Exception as exc:
            err = _error_html(str(exc))
            return err, err, err, err, None

    # ── handler: Compare ────────────────────────────────────────────

    def _handle_compare(image_a, image_b):
        """Classify both images and compute cosine similarity."""
        try:
            from artsleuth.config import AnalysisConfig
            from artsleuth.core.style import StyleClassifier

            if image_a is None or image_b is None:
                msg = _info_html("Upload two artworks to compare.")
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
                    "Very high embedding similarity \u2014 the works share "
                    "strong stylistic features. This does not confirm "
                    "shared authorship on its own."
                )
            elif similarity > 0.65:
                interp = (
                    "Moderate similarity \u2014 some shared stylistic "
                    "traits, but distinct in other respects."
                )
            elif similarity > 0.40:
                interp = (
                    "Low similarity \u2014 different stylistic profiles "
                    "with some overlapping features."
                )
            else:
                interp = (
                    "Minimal similarity \u2014 markedly different "
                    "stylistic profiles."
                )

            sim_html = (
                '<div class="as-card">'
                '<div class="as-sim">'
                f'<div class="as-sim__value">{similarity:.1%}</div>'
                '<div class="as-sim__label">Embedding Similarity</div>'
                f'<div class="as-sim__interp">{interp}</div>'
                "</div></div>"
            )

            def _axis_row(label, pred_a, pred_b):
                return (
                    f"<tr><td><strong>{label}</strong></td>"
                    f"<td>{pred_a.label} ({pred_a.confidence:.0%})</td>"
                    f"<td>{pred_b.label} ({pred_b.confidence:.0%})</td>"
                    "</tr>"
                )

            comparison_html = (
                '<div class="as-card" style="margin-top:0.5rem">'
                '<h3 class="as-card__title">Style Comparison</h3>'
                '<table class="as-table">'
                "<thead><tr>"
                "<th>Axis</th><th>Artwork A</th><th>Artwork B</th>"
                "</tr></thead><tbody>"
                + _axis_row("Period", report_a.period, report_b.period)
                + _axis_row("School", report_a.school, report_b.school)
                + _axis_row(
                    "Genre", report_a.technique, report_b.technique,
                )
                + "</tbody></table></div>"
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
                        "Try a higher-resolution image.",
                    ),
                    None,
                )

            embeddings = np.stack(
                [d.embedding for d in brushstroke_report.descriptors],
            )
            bboxes = [
                d.bbox for d in brushstroke_report.descriptors
            ]
            coherences = np.array(
                [d.coherence for d in brushstroke_report.descriptors],
            )
            energies = np.array(
                [d.energy for d in brushstroke_report.descriptors],
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
            from artsleuth.core.temporal import (
                TemporalRegistry,
                estimate_date_from_style,
            )

            if image is None:
                return _info_html(
                    "Upload an artwork to estimate its date.",
                )

            config = AnalysisConfig()
            classifier = StyleClassifier(config)
            style_report = classifier.classify(image)

            prediction = None
            method = "style classification"

            if artist_name.strip():
                registry = TemporalRegistry()
                prediction = registry.predict(
                    artist_name.strip(), style_report.embedding,
                )
                if prediction is not None:
                    method = (
                        f"temporal model for {artist_name.strip()}"
                    )

            if prediction is None:
                prediction = estimate_date_from_style(
                    style_report.period.top_k,
                )
                method = (
                    f"style classification "
                    f"({style_report.period.label})"
                )

            lo, hi = prediction.confidence_band
            score_cls = (
                "as-date__stat-value--good"
                if prediction.temporal_score > 0.7
                else "as-date__stat-value--mid"
                if prediction.temporal_score > 0.4
                else "as-date__stat-value--bad"
            )
            method_safe = html_module.escape(method)

            return (
                '<div class="as-card">'
                '<h3 class="as-card__title">Temporal Estimation</h3>'
                '<div class="as-date">'
                f'<div class="as-date__year">'
                f"c.\u2009{prediction.estimated_year:.0f}</div>"
                '<div class="as-date__label">Estimated Date</div>'
                '<div class="as-date__stats">'
                "<div>"
                '<div class="as-date__stat-label">95% Band</div>'
                '<div class="as-date__stat-value">'
                f"{lo:.0f}\u2013{hi:.0f}</div>"
                "</div>"
                "<div>"
                '<div class="as-date__stat-label">Plausibility</div>'
                f'<div class="as-date__stat-value {score_cls}">'
                f"{prediction.temporal_score:.0%}</div>"
                "</div>"
                "<div>"
                '<div class="as-date__stat-label">'
                "Drift / Decade</div>"
                '<div class="as-date__stat-value">'
                f"{prediction.drift_rate:.3f}</div>"
                "</div></div>"
                f'<div class="as-date__foot">'
                f"Based on {method_safe}</div>"
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
            # ── Tab 1: Analyse ──────────────────────────────────
            with gr.Tab("Analyse"):
                with gr.Row():
                    with gr.Column(scale=1):
                        analyse_image = gr.Image(
                            type="pil", label="Upload Artwork",
                        )
                        analyse_ref = gr.Textbox(
                            label="Reference Artist (optional)",
                            placeholder="e.g. Rembrandt",
                        )
                        analyse_btn = gr.Button(
                            "Analyse", variant="primary",
                        )
                    with gr.Column(scale=2):
                        analyse_style = gr.HTML(
                            label="Style Report",
                        )
                        analyse_artist = gr.HTML(
                            label="Artist Estimation",
                        )
                        analyse_attr = gr.HTML(
                            label="Attribution Report",
                        )
                        analyse_forgery = gr.HTML(
                            label="Forgery Screening",
                        )
                        analyse_heatmap = gr.Image(
                            label="Saliency Heatmap",
                        )

                analyse_btn.click(
                    fn=_handle_analyse,
                    inputs=[analyse_image, analyse_ref],
                    outputs=[
                        analyse_style,
                        analyse_artist,
                        analyse_attr,
                        analyse_forgery,
                        analyse_heatmap,
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

            # ── Tab 4: Estimate Date ────────────────────────────
            with gr.Tab("Estimate Date"):
                with gr.Row():
                    with gr.Column(scale=1):
                        tl_image = gr.Image(
                            type="pil", label="Upload Artwork",
                        )
                        tl_artist = gr.Textbox(
                            label="Artist Name (optional)",
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


def _format_artist_estimation(
    candidates: list[tuple[str, float]],
) -> str:
    """Render zero-shot artist candidates as HTML."""
    if not candidates:
        return _info_html(
            "Could not estimate artist for this image.",
        )

    top_name, top_conf = candidates[0]
    bars = ""
    for idx, (name, conf) in enumerate(candidates):
        pct = min(max(conf * 100, 0), 100)
        variant = "gold" if idx == 0 else "amber"
        bars += (
            f'<div class="as-bar as-bar--{variant}">'
            f'<div class="as-bar__header">'
            f"<span>{html_module.escape(name)}</span>"
            f'<span class="as-bar__pct">{pct:.1f}%</span></div>'
            f'<div class="as-bar__track">'
            f'<div class="as-bar__fill" style="width:{pct:.1f}%">'
            f"</div></div></div>"
        )

    return (
        '<div class="as-card">'
        '<h3 class="as-card__title">Artist Estimation</h3>'
        f'<div class="as-consensus">'
        f"<strong>Most likely:</strong> "
        f"{html_module.escape(top_name)} ({top_conf:.0%})</div>"
        f"{bars}"
        '<div class="as-note">'
        "CLIP zero-shot estimation. "
        "For definitive attribution, consult a qualified "
        "art historian.</div>"
        "</div>"
    )


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
    rows_data = [
        ("DINOv2 &middot; ViT-B/14", "57.5%", "0.553",
         "64.7%", "90.9%", "71.0%", False),
        ("CLIP &middot; ViT-L/14", "67.1%", "0.656",
         "74.6%", "95.9%", "75.0%", False),
        ("Fusion &middot; frozen", "65.0%", "0.633",
         "71.0%", "94.2%", "74.2%", False),
        ("Fusion &middot; fine-tuned", "71.6%", "0.703",
         "77.8%", "96.2%", "75.1%", False),
        ("Fusion &middot; end-to-end", "72.7%", "&mdash;",
         "79.0%", "96.9%", "76.6%", True),
    ]

    rows_html = ""
    for name, style, f1, artist, top5, genre, highlight in rows_data:
        hl = ' class="as-table--highlight"' if highlight else ""
        rows_html += (
            f"<tr>"
            f"<td{hl}>{name}</td>"
            f"<td{hl}>{style}</td>"
            f"<td{hl}>{f1}</td>"
            f"<td{hl}>{artist}</td>"
            f"<td{hl}>{top5}</td>"
            f"<td{hl}>{genre}</td>"
            f"</tr>"
        )

    return (
        '<div class="as-bench">'
        '<h3 class="as-bench__title">WikiArt Benchmark</h3>'
        '<div class="as-bench__sub">'
        "81,444 images &middot; 27 styles &middot; "
        "129 artists &middot; 11 genres</div>"
        "</div>"
        '<table class="as-table">'
        "<thead><tr>"
        "<th>Backbone</th><th>Style</th><th>F1</th>"
        "<th>Artist</th><th>Top-5</th><th>Genre</th>"
        "</tr></thead><tbody>"
        + rows_html
        + "</tbody></table>"
        '<p class="as-note" style="text-align:center;margin-top:1rem">'
        "Top four rows: linear probes. Bottom row: end-to-end heads. "
        "Fine-tuning: SupCon + CE, 3-block unfreeze, cosine annealing, "
        "5 epochs, Tesla P100. All numbers macro-averaged.</p>"
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
