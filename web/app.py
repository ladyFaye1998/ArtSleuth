"""ArtSleuth web demo — dark gallery edition.

Interactive art analysis powered by vision transformers. Upload a
painting and get brushstroke analysis, style classification, artist
attribution, forgery screening, and workshop decomposition.
"""
from __future__ import annotations

import html as html_module
import tempfile

import gradio as gr
import numpy as np

from web.theme import HTML_ELEM_CLASSES


# ── Callout helpers ─────────────────────────────────────────────────

def _error_html(message: str) -> str:
    safe = html_module.escape(str(message))
    return f'<div class="as-msg as-msg--error"><strong>Error:</strong> {safe}</div>'


def _info_html(message: str) -> str:
    safe = html_module.escape(str(message))
    return f'<div class="as-msg as-msg--info">{safe}</div>'


def _save_pil_to_temp(image) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp, format="PNG")
        return tmp.name


# ── Card wrapper used for inline results ────────────────────────────

def _card(title: str, body: str) -> str:
    return (
        '<div class="as-card">'
        f"<h3>{html_module.escape(title)}</h3>"
        f"{body}"
        "</div>"
    )


def _bar(label: str, value: float, color: str = "#d4a843") -> str:
    pct = min(max(value * 100, 0), 100)
    return (
        '<div class="as-bar-wrap">'
        '<div class="as-bar-top">'
        f"<span>{html_module.escape(label)}</span>"
        f'<span class="as-pct">{pct:.1f}%</span>'
        "</div>"
        '<div class="as-bar-track">'
        f'<div class="as-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>'
        "</div></div>"
    )


# ── UI builder ──────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    from web.theme import artsleuth_theme, CUSTOM_CSS, HEADER_HTML, FOOTER_HTML

    # ── handler: Analyse ────────────────────────────────────────

    def _handle_analyse(image, reference_artist: str):
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
            result = run_pipeline(path, config=config, reference_artist=ref)

            style_html = format_style_report(result.style)

            try:
                classifier = StyleClassifier(config)
                artist_top = classifier.estimate_artist(
                    result.style.embedding, top_k=5,
                )
            except Exception:
                artist_top = []
            artist_html = _format_artist_estimation(artist_top)

            attribution_html = format_attribution_report(result.attribution)

            if result.forgery is not None:
                forgery_html = format_forgery_gauge(result.forgery)
            else:
                forgery_html = _info_html(
                    "Provide a reference artist to enable forgery screening."
                )

            heatmap_image = None
            try:
                explanation = result.explain(target="attribution")
                if explanation.composite is not None:
                    heatmap_image = explanation.composite
            except Exception:
                pass

            return style_html, artist_html, attribution_html, forgery_html, heatmap_image
        except Exception as exc:
            err = _error_html(str(exc))
            return err, err, err, err, None

    # ── handler: Compare ────────────────────────────────────────

    def _handle_compare(image_a, image_b):
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

            emb_a, emb_b = report_a.embedding, report_b.embedding
            similarity = float(np.dot(emb_a, emb_b)) / (
                float(np.linalg.norm(emb_a)) * float(np.linalg.norm(emb_b)) + 1e-12
            )

            if similarity > 0.85:
                interp = "Very high — strong shared stylistic features."
            elif similarity > 0.65:
                interp = "Moderate — some shared traits, distinct in other respects."
            elif similarity > 0.40:
                interp = "Low — different profiles with some overlap."
            else:
                interp = "Minimal — markedly different styles."

            sim_html = _card("Embedding Similarity", (
                '<div style="text-align:center;padding:1rem 0;">'
                '<div style="font-family:\'Playfair Display\',serif;'
                f'font-size:3.2rem;font-weight:700;color:#faf9f6;">{similarity:.1%}</div>'
                '<div style="font-size:0.7rem;letter-spacing:0.1em;'
                'text-transform:uppercase;color:#d4a843;font-weight:600;'
                'margin-top:0.2rem;">Cosine Similarity</div>'
                f'<div style="font-size:0.88rem;color:#a8a29e;'
                f'margin-top:0.8rem;max-width:420px;margin-left:auto;'
                f'margin-right:auto;">{interp}</div></div>'
            ))

            def _row(label, pa, pb):
                return (
                    f"<tr><td style='padding:6px 10px;font-weight:500;'>"
                    f"<strong>{label}</strong></td>"
                    f"<td style='padding:6px 10px;'>{pa.label} ({pa.confidence:.0%})</td>"
                    f"<td style='padding:6px 10px;'>{pb.label} ({pb.confidence:.0%})</td>"
                    "</tr>"
                )

            table_html = _card("Style Comparison", (
                '<table style="width:100%;border-collapse:collapse;font-size:0.88rem;">'
                '<thead><tr style="border-bottom:1px solid rgba(212,168,67,0.2);">'
                '<th style="padding:6px 10px;text-align:left;">Axis</th>'
                '<th style="padding:6px 10px;">Artwork A</th>'
                '<th style="padding:6px 10px;">Artwork B</th></tr></thead><tbody>'
                + _row("Period", report_a.period, report_b.period)
                + _row("School", report_a.school, report_b.school)
                + _row("Genre", report_a.technique, report_b.technique)
                + "</tbody></table>"
            ))
            return sim_html, table_html
        except Exception as exc:
            err = _error_html(str(exc))
            return err, err

    # ── handler: Workshop ───────────────────────────────────────

    def _handle_workshop(image, max_hands: int):
        try:
            from artsleuth.config import AnalysisConfig
            from artsleuth.core.brushstroke import BrushstrokeAnalyzer
            from artsleuth.core.workshop import WorkshopDecomposition
            from web.components import format_workshop_report

            if image is None:
                return _info_html("Upload an artwork to decompose."), None

            config = AnalysisConfig(workshop_max_hands=int(max_hands))
            analyzer = BrushstrokeAnalyzer(config)
            brushstroke_report = analyzer.analyze(image)

            if not brushstroke_report.descriptors:
                return _info_html("No brushstroke patches extracted. Try higher resolution."), None

            embeddings = np.stack([d.embedding for d in brushstroke_report.descriptors])
            bboxes = [d.bbox for d in brushstroke_report.descriptors]
            coherences = np.array([d.coherence for d in brushstroke_report.descriptors])
            energies = np.array([d.energy for d in brushstroke_report.descriptors])

            decomposer = WorkshopDecomposition(max_hands=int(max_hands))
            workshop_report = decomposer.decompose(
                embeddings, bboxes, image.size,
                coherences=coherences, energies=energies,
            )

            report_html = format_workshop_report(workshop_report)
            hand_map_image = None
            if workshop_report.hand_map is not None:
                unique_ids = np.unique(workshop_report.hand_map)
                valid_ids = unique_ids[unique_ids >= 0]
                palette = _hand_map_palette(len(valid_ids))

                h, w = workshop_report.hand_map.shape
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                for hid in valid_ids:
                    mask = workshop_report.hand_map == hid
                    canvas[mask] = palette[int(hid) % len(palette)]

                orig = np.array(image.convert("RGB"))
                if orig.shape[:2] == (h, w):
                    blended = (0.6 * orig.astype(np.float32) + 0.4 * canvas.astype(np.float32))
                    hand_map_image = blended.astype(np.uint8)
                else:
                    hand_map_image = canvas

            return report_html, hand_map_image
        except Exception as exc:
            return _error_html(str(exc)), None

    # ── handler: Timeline ───────────────────────────────────────

    def _handle_timeline(image, artist_name: str):
        try:
            from artsleuth.config import AnalysisConfig
            from artsleuth.core.style import StyleClassifier
            from artsleuth.core.temporal import TemporalRegistry, estimate_date_from_style

            if image is None:
                return _info_html("Upload an artwork to estimate its date.")

            config = AnalysisConfig()
            classifier = StyleClassifier(config)
            style_report = classifier.classify(image)

            prediction = None
            method = "style classification"

            if artist_name.strip():
                registry = TemporalRegistry()
                prediction = registry.predict(artist_name.strip(), style_report.embedding)
                if prediction is not None:
                    method = f"temporal model for {artist_name.strip()}"

            if prediction is None:
                prediction = estimate_date_from_style(style_report.period.top_k)
                method = f"style classification ({style_report.period.label})"

            lo, hi = prediction.confidence_band
            score_cls = (
                "as-score-good" if prediction.temporal_score > 0.7
                else "as-score-mid" if prediction.temporal_score > 0.4
                else "as-score-bad"
            )
            method_safe = html_module.escape(method)

            return _card("Temporal Estimation", (
                '<div class="as-date">'
                f'<div class="as-date__year">c.\u2009{prediction.estimated_year:.0f}</div>'
                '<div class="as-date__label">Estimated Date</div>'
                '<div class="as-date__stats">'
                '<div style="text-align:center">'
                '<div class="as-date__stat-label">95% Band</div>'
                f'<div class="as-date__stat-value">{lo:.0f}\u2013{hi:.0f}</div>'
                "</div>"
                '<div style="text-align:center">'
                '<div class="as-date__stat-label">Plausibility</div>'
                f'<div class="as-date__stat-value {score_cls}">'
                f"{prediction.temporal_score:.0%}</div>"
                "</div>"
                '<div style="text-align:center">'
                '<div class="as-date__stat-label">Drift / Decade</div>'
                f'<div class="as-date__stat-value">{prediction.drift_rate:.3f}</div>'
                "</div></div>"
                f'<div class="as-date__foot">Based on {method_safe}</div>'
                "</div>"
            ))
        except Exception as exc:
            return _error_html(str(exc))

    # ── layout ──────────────────────────────────────────────────

    with gr.Blocks(theme=artsleuth_theme(), css=CUSTOM_CSS) as app:
        gr.HTML(HEADER_HTML)

        with gr.Tabs():
            with gr.Tab("Analyse"):
                with gr.Row():
                    with gr.Column(scale=1):
                        analyse_image = gr.Image(type="pil", label="Upload Artwork")
                        analyse_ref = gr.Textbox(
                            label="Reference Artist (optional)",
                            placeholder="e.g. Rembrandt",
                        )
                        analyse_btn = gr.Button("Analyse", variant="primary")
                    with gr.Column(scale=2):
                        analyse_style = gr.HTML(label="Style", elem_classes=HTML_ELEM_CLASSES)
                        analyse_artist = gr.HTML(label="Artist", elem_classes=HTML_ELEM_CLASSES)
                        analyse_attr = gr.HTML(label="Attribution", elem_classes=HTML_ELEM_CLASSES)
                        analyse_forgery = gr.HTML(label="Forgery", elem_classes=HTML_ELEM_CLASSES)
                        analyse_heatmap = gr.Image(label="Saliency Heatmap")

                analyse_btn.click(
                    fn=_handle_analyse,
                    inputs=[analyse_image, analyse_ref],
                    outputs=[analyse_style, analyse_artist, analyse_attr,
                             analyse_forgery, analyse_heatmap],
                )

            with gr.Tab("Compare"):
                with gr.Row():
                    compare_a = gr.Image(type="pil", label="Artwork A")
                    compare_b = gr.Image(type="pil", label="Artwork B")
                compare_btn = gr.Button("Compare", variant="primary")
                compare_sim = gr.HTML(label="Similarity", elem_classes=HTML_ELEM_CLASSES)
                compare_axes = gr.HTML(label="Comparison", elem_classes=HTML_ELEM_CLASSES)
                compare_btn.click(
                    fn=_handle_compare,
                    inputs=[compare_a, compare_b],
                    outputs=[compare_sim, compare_axes],
                )

            with gr.Tab("Workshop"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ws_image = gr.Image(type="pil", label="Upload Artwork")
                        ws_hands = gr.Slider(minimum=2, maximum=8, value=6, step=1, label="Max Hands")
                        ws_btn = gr.Button("Decompose", variant="primary")
                    with gr.Column(scale=2):
                        ws_report = gr.HTML(label="Report", elem_classes=HTML_ELEM_CLASSES)
                        ws_map = gr.Image(label="Hand Map Overlay")
                ws_btn.click(
                    fn=_handle_workshop,
                    inputs=[ws_image, ws_hands],
                    outputs=[ws_report, ws_map],
                )

            with gr.Tab("Estimate Date"):
                with gr.Row():
                    with gr.Column(scale=1):
                        tl_image = gr.Image(type="pil", label="Upload Artwork")
                        tl_artist = gr.Textbox(
                            label="Artist Name (optional)",
                            placeholder="e.g. Artemisia Gentileschi",
                        )
                        tl_btn = gr.Button("Estimate Date", variant="primary")
                    with gr.Column(scale=2):
                        tl_result = gr.HTML(label="Prediction", elem_classes=HTML_ELEM_CLASSES)
                tl_btn.click(
                    fn=_handle_timeline,
                    inputs=[tl_image, tl_artist],
                    outputs=[tl_result],
                )

            with gr.Tab("Benchmark"):
                gr.HTML(_build_benchmark_table(), elem_classes=HTML_ELEM_CLASSES)
                gr.Markdown(_benchmark_methodology())

        gr.HTML(FOOTER_HTML)

    return app


# ── helpers ─────────────────────────────────────────────────────────

def _format_artist_estimation(candidates: list[tuple[str, float]]) -> str:
    if not candidates:
        return _info_html("Could not estimate artist for this image.")

    top_name, top_conf = candidates[0]
    bars = "".join(
        _bar(name, conf, "#d4a843") for name, conf in candidates
    )
    return _card("Artist Estimation", (
        '<div style="font-size:0.92rem;margin-bottom:0.6rem;">'
        f"<strong>Most likely:</strong> {html_module.escape(top_name)} ({top_conf:.0%})"
        "</div>"
        f"{bars}"
        '<div style="font-size:0.74rem;color:#78716c;margin-top:0.6rem;font-style:italic;">'
        "CLIP zero-shot estimation. "
        "For definitive attribution, consult a qualified art historian."
        "</div>"
    ))


def _hand_map_palette(n: int) -> list[list[int]]:
    base = [
        [212, 168, 67], [240, 118, 138], [104, 181, 213],
        [74, 222, 128], [167, 139, 250], [251, 191, 36],
        [220, 170, 120], [100, 200, 180],
    ]
    return base[:max(n, 1)]


def _build_benchmark_table() -> str:
    def _row(name, style, f1, artist, top5, genre, highlight=False):
        cls = ' style="color:#d4a843;font-weight:600;"' if highlight else ""
        return (
            f"<tr{cls}>"
            f'<td style="padding:0.5rem 0.8rem;">{name}</td>'
            f'<td style="padding:0.5rem 0.8rem;text-align:center;">{style}</td>'
            f'<td style="padding:0.5rem 0.8rem;text-align:center;">{f1}</td>'
            f'<td style="padding:0.5rem 0.8rem;text-align:center;">{artist}</td>'
            f'<td style="padding:0.5rem 0.8rem;text-align:center;">{top5}</td>'
            f'<td style="padding:0.5rem 0.8rem;text-align:center;">{genre}</td>'
            "</tr>"
        )

    rows = (
        _row("DINOv2 &middot; ViT-B/14", "57.5%", "0.553", "64.7%", "90.9%", "71.0%")
        + _row("CLIP &middot; ViT-L/14", "67.1%", "0.656", "74.6%", "95.9%", "75.0%")
        + _row("Fusion &middot; frozen", "65.0%", "0.633", "71.0%", "94.2%", "74.2%")
        + _row("Fusion &middot; fine-tuned", "71.6%", "0.703", "77.8%", "96.2%", "75.1%")
        + _row("Fusion &middot; end-to-end", "72.7%", "&mdash;", "79.0%", "96.9%", "76.6%",
               highlight=True)
    )

    return _card("WikiArt Benchmark", (
        '<div style="text-align:center;margin-bottom:1rem;">'
        '<div style="font-size:0.76rem;color:#78716c;">'
        "81,444 images &middot; 27 styles &middot; 129 artists &middot; 11 genres"
        "</div></div>"
        '<table style="width:100%;border-collapse:collapse;font-size:0.84rem;">'
        '<thead><tr>'
        '<th style="padding:0.5rem 0.8rem;text-align:left;">Backbone</th>'
        '<th style="padding:0.5rem 0.8rem;text-align:center;">Style</th>'
        '<th style="padding:0.5rem 0.8rem;text-align:center;">F1</th>'
        '<th style="padding:0.5rem 0.8rem;text-align:center;">Artist</th>'
        '<th style="padding:0.5rem 0.8rem;text-align:center;">Top-5</th>'
        '<th style="padding:0.5rem 0.8rem;text-align:center;">Genre</th>'
        "</tr></thead><tbody>"
        + rows
        + "</tbody></table>"
        '<div style="font-size:0.72rem;color:#57534e;margin-top:0.8rem;text-align:center;">'
        "Linear probes (top 4), end-to-end heads (bottom). "
        "SupCon + CE, cosine annealing, 5 epochs, Tesla P100. Macro-averaged."
        "</div>"
    ))


def _benchmark_methodology() -> str:
    return (
        "### Methodology\n\n"
        "All models evaluated on the full **WikiArt** dataset "
        "(81,444 images, 80/20 split, 27 styles, 129 artists, 11 genres). "
        "Style and genre accuracy use single-label top-1 matching; "
        "artist accuracy reports both top-1 and top-5.\n\n"
        "**ArtSleuth Fusion** combines DINOv2 ViT-B/14 texture features "
        "with CLIP ViT-L/14 semantic embeddings via a learnable "
        "cross-attention fusion head.\n\n"
        "Training uses AdamW with cosine annealing (backbone lr=1e-5, "
        "head lr=5e-4), mixed precision, gradient accumulation (effective "
        "batch 64), for 5 epochs on a Tesla P100.\n\n"
        "*All numbers are macro-averaged across classes.*"
    )


if __name__ == "__main__":
    app = create_app()
    app.launch()
