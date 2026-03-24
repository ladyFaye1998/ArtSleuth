"""Custom Gradio theme for ArtSleuth.

Defines a cohesive visual identity using the ArtSleuth design system:
navy primary, soft-blue secondary, rose accent, gold tertiary, cream
neutrals, set in Cormorant Garamond for display and Inter for body.
"""
from __future__ import annotations

import gradio as gr

NAVY = "#0f1f35"
NAVY_MID = "#162d4a"
SOFT_BLUE = "#7fb3d3"
ROSE = "#c27889"
GOLD = "#c9a84c"
CREAM = "#f4efe8"
CREAM_DARK = "#e8e0d4"
TEXT_LIGHT = "#ddd8d0"
WHITE = "#ffffff"
MUTED = "#6b5e50"

# --- gr.HTML surfaces -------------------------------------------------------
# HuggingFace Spaces (and some browser / Gradio skins) use a dark chrome
# around ``gr.HTML`` while our copy assumes a light card.  Use explicit
# light panels + high-contrast text so nothing sits navy-on-near-black.
HTML_LIGHT_PANEL = (
    "background:linear-gradient(180deg,#fcfaf7 0%,#f2ebe3 100%);"
    "color:#0f1f35;"
    "padding:1.4rem 1.6rem;"
    "border-radius:12px;"
    "border:1px solid rgba(127,179,211,0.55);"
    "box-shadow:0 4px 24px rgba(0,0,0,0.14);"
)
HTML_TEXT_MUTED = "#3d3834"
HTML_INFO_BOX = (
    "background:#e8f0f6;"
    "color:#0f1f35;"
    "border:1px solid rgba(127,179,211,0.65);"
    "border-radius:10px;"
    "padding:1rem 1.2rem;"
    "margin:0.5rem 0;"
    "font-family:'Inter',sans-serif;"
    "font-size:0.88rem;"
    "box-shadow:0 2px 12px rgba(0,0,0,0.08);"
)
HTML_ERROR_BOX = (
    "color:#6b1f2e;"
    "background:#fce8ec;"
    "border:1px solid rgba(194,120,137,0.45);"
    "border-radius:10px;"
    "padding:1rem 1.2rem;"
    "margin:0.5rem 0;"
    "font-family:'Inter',sans-serif;"
    "font-size:0.88rem;"
    "box-shadow:0 2px 12px rgba(0,0,0,0.08);"
)

# Apply to ``gr.HTML(..., elem_classes=HTML_ELEM_CLASSES)`` so global CSS can
# beat Gradio/Tailwind ``.prose`` dark text on dark block backgrounds.
HTML_ELEM_CLASSES: list[str] = ["artsleuth-html-host"]


def artsleuth_theme() -> gr.themes.Base:
    """Build and return the ArtSleuth Gradio theme."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e4eaf1", c100="#c2d0e0", c200="#99b3cd",
            c300="#7096ba", c400="#4779a7", c500=NAVY,
            c600="#0d1b2f", c700="#0b1626", c800="#08101d",
            c900="#060b14", c950="#03060b",
        ),
        secondary_hue=gr.themes.Color(
            c50="#eef5f9", c100="#d6e8f1", c200="#bddbe9",
            c300=SOFT_BLUE, c400="#5fa0c0", c500="#4090b3",
            c600="#357fa0", c700="#2c6d8a", c800="#235b73",
            c900="#1a495c", c950="#123745",
        ),
        neutral_hue=gr.themes.Color(
            c50=CREAM, c100=CREAM_DARK, c200="#dcd4c8",
            c300="#c8bdb2", c400="#b0a498", c500="#988b7e",
            c600="#807264", c700=MUTED, c800="#504030",
            c900="#382816", c950="#201000",
        ),
        font=[
            gr.themes.GoogleFont("Inter"),
            "-apple-system",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("Fira Code"),
            "Consolas",
            "monospace",
        ],
    ).set(
        body_background_fill=CREAM,
        body_text_color=NAVY,
        button_primary_background_fill=GOLD,
        button_primary_background_fill_hover="#b89540",
        button_primary_text_color=NAVY,
        button_primary_border_color=GOLD,
        button_secondary_background_fill=SOFT_BLUE,
        button_secondary_background_fill_hover="#6aa3c4",
        button_secondary_text_color=NAVY,
        button_secondary_border_color=SOFT_BLUE,
        block_title_text_color=NAVY,
        block_label_text_color=NAVY,
        block_background_fill=WHITE,
        block_border_color="rgba(127,179,211,0.2)",
        block_border_width="1px",
        block_shadow="0 2px 16px rgba(15,31,53,0.06)",
        input_background_fill=WHITE,
        input_border_color="rgba(127,179,211,0.25)",
        input_placeholder_color="#988b7e",
        checkbox_label_background_fill=CREAM,
        checkbox_background_color=WHITE,
        checkbox_border_color=SOFT_BLUE,
        border_color_accent=GOLD,
        color_accent=GOLD,
        link_text_color=ROSE,
        link_text_color_hover=GOLD,
        shadow_spread="4px",
        block_radius="10px",
        table_border_color="rgba(127,179,211,0.2)",
        table_even_background_fill=CREAM,
        table_odd_background_fill=WHITE,
    )
    return theme


HEADER_HTML = """
<div class="artsleuth-header">
  <h1>ArtSleuth</h1>
  <p>Computational Art Analysis Framework</p>
  <div class="header-line"></div>
</div>
"""

FOOTER_HTML = """
<div class="artsleuth-footer">
  <div class="footer-line"></div>
  ArtSleuth &mdash; Computational Art Analysis Framework &bull; Research use only<br>
  <span class="footer-sub">
    Results are probabilistic and should be reviewed by qualified art historians.
    &ensp;|&ensp;
    <a href="https://github.com/ladyFaye1998/ArtSleuth" target="_blank">GitHub</a>
    &ensp;|&ensp;
    <a href="https://huggingface.co/ladyFaye1998/artsleuth-weights" target="_blank">Weights</a>
  </span>
</div>
"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

.gradio-container {
    font-family: 'Inter', -apple-system, sans-serif !important;
    max-width: 1200px;
    margin: 0 auto;
}

/* Header */
.artsleuth-header {
    background: linear-gradient(145deg, #0f1f35 0%, #162d4a 50%, #1e3a5f 100%);
    color: #f4efe8;
    text-align: center;
    padding: 2.5rem 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.artsleuth-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 300px 300px at 25% 60%, rgba(201,168,76,0.06), transparent),
        radial-gradient(ellipse 250px 250px at 75% 30%, rgba(127,179,211,0.06), transparent);
}
.artsleuth-header h1 {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0 0 0.2rem;
    letter-spacing: 0.06em;
    background: linear-gradient(135deg, #c9a84c, #e6c96e, #c9a84c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
}
.artsleuth-header p {
    font-size: 1rem;
    margin: 0;
    color: #ddd8d0;
    font-style: italic;
    font-weight: 300;
    position: relative;
    opacity: 0.8;
}
.header-line {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #c9a84c, transparent);
    margin: 1rem auto 0;
    position: relative;
}

/* Tabs */
.tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
    color: #0f1f35 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}
.tab-nav button:hover {
    color: #7fb3d3 !important;
    border-bottom-color: #7fb3d3 !important;
}
.tab-nav button.selected {
    color: #0f1f35 !important;
    border-bottom: 3px solid #c9a84c !important;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, #c9a84c, #b8953f) !important;
    color: #0f1f35 !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(201,168,76,0.25) !important;
    transition: all 0.3s !important;
}
button.primary:hover {
    box-shadow: 0 6px 24px rgba(201,168,76,0.4) !important;
    transform: translateY(-1px) !important;
}

/* Footer */
.artsleuth-footer {
    text-align: center;
    padding: 1.5rem 1rem;
    margin-top: 1.5rem;
    color: #6b5e50;
    font-size: 0.82rem;
    font-weight: 400;
}
.footer-line {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #7fb3d3, transparent);
    margin: 0 auto 1rem;
}
.footer-sub {
    font-size: 0.75rem;
    color: #988b7e;
    display: block;
    margin-top: 0.4rem;
}
.footer-sub a {
    color: #c27889;
    text-decoration: none;
}
.footer-sub a:hover {
    color: #c9a84c;
}

/* Tables in benchmark tab */
table {
    font-size: 0.88rem !important;
}
table th {
    background: #0f1f35 !important;
    color: #f4efe8 !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
}
table td {
    border-bottom: 1px solid rgba(127,179,211,0.15) !important;
}

/* Gauge and result cards */
.result-section {
    background: #ffffff;
    border: 1px solid rgba(127,179,211,0.15);
    border-radius: 10px;
    padding: 1.2rem;
}

/* -------------------------------------------------------------------------
   gr.HTML on HF Spaces: Gradio wraps content in ``.prose`` and applies
   theme/dark text colours. Inline ``style=`` on inner divs is often
   overridden.  Force a light panel on the component host and reset prose.
   ------------------------------------------------------------------------- */
.artsleuth-html-host {
    background-color: #e8dfd4 !important;
    background-image: none !important;
    color: #0a1422 !important;
    border-radius: 12px !important;
    padding: 12px !important;
    border: 1px solid rgba(100, 130, 165, 0.4) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.35) !important;
}

.artsleuth-html-host .prose,
.artsleuth-html-host .prose-sm {
    color: #0a1422 !important;
}

/* Inline callouts (forgery hint, upload prompts, etc.) */
.artsleuth-html-host .prose .artsleuth-msg--info,
.artsleuth-html-host .artsleuth-msg--info {
    display: block !important;
    background: #cfe4f2 !important;
    color: #041018 !important;
    border: 1px solid #6a9eb8 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    margin: 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.45 !important;
    font-weight: 500 !important;
}

.artsleuth-html-host .prose .artsleuth-msg--error,
.artsleuth-html-host .artsleuth-msg--error {
    display: block !important;
    background: #fcd4dc !important;
    color: #4a0a14 !important;
    border: 1px solid #c27889 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    margin: 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}

/* Date panel (Estimate Date tab): large figures must stay legible */
.artsleuth-html-host .prose .artsleuth-date-panel,
.artsleuth-html-host .artsleuth-date-panel {
    text-align: center !important;
    background: linear-gradient(180deg, #fffdf9 0%, #f2ebe3 100%) !important;
    color: #0a1422 !important;
    border: 1px solid rgba(127,179,211,0.5) !important;
    border-radius: 12px !important;
    padding: 1.5rem 1.25rem !important;
    margin: 0 !important;
}

.artsleuth-html-host .artsleuth-date-panel .artsleuth-date-figure {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    color: #061018 !important;
    line-height: 1.1 !important;
}

.artsleuth-html-host .artsleuth-date-panel .artsleuth-gold-label {
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #6b5610 !important;
    font-weight: 600 !important;
    margin-top: 0.25rem !important;
}

.artsleuth-html-host .artsleuth-date-panel .artsleuth-muted {
    font-size: 0.68rem !important;
    color: #3d3834 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

.artsleuth-html-host .artsleuth-date-panel .artsleuth-date-sub {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #061018 !important;
    margin-top: 2px !important;
}

.artsleuth-html-host .artsleuth-date-panel .artsleuth-date-sub.artsleuth-score-good {
    color: #1b5e20 !important;
}
.artsleuth-html-host .artsleuth-date-panel .artsleuth-date-sub.artsleuth-score-mid {
    color: #e65100 !important;
}
.artsleuth-html-host .artsleuth-date-panel .artsleuth-date-sub.artsleuth-score-bad {
    color: #b71c1c !important;
}

.artsleuth-html-host .artsleuth-date-panel .artsleuth-date-foot {
    font-size: 0.78rem !important;
    color: #3d3834 !important;
    margin-top: 1rem !important;
    font-style: italic !important;
}
"""
