"""Custom Gradio theme for ArtSleuth.

Dark-native warm editorial design — like a dimly lit museum gallery.
Rich charcoal backgrounds, warm amber/gold accents, elegant serif
display type, generous spacing, quiet confidence.

All component HTML uses CSS classes defined in CUSTOM_CSS rather than
inline ``style=`` attributes, so Gradio's ``.prose`` wrapper and
dark-mode rendering never fight the design.
"""
from __future__ import annotations

import gradio as gr

# ---------------------------------------------------------------------------
# Design tokens  (referenced by CUSTOM_CSS as --as-* custom properties)
# ---------------------------------------------------------------------------

BG = "#1a1714"
SURFACE = "#242019"
SURFACE_RAISED = "#2e2923"
BORDER = "rgba(201,168,76,0.12)"
BORDER_STRONG = "rgba(201,168,76,0.25)"

TEXT = "#e8e0d4"
TEXT_MUTED = "#a89f94"
TEXT_DIM = "#6e655c"

GOLD = "#c9a84c"
GOLD_DIM = "#8a7434"
AMBER = "#d4956a"
ROSE = "#c27889"
GREEN = "#7dab6c"
RED = "#c2555a"


def artsleuth_theme() -> gr.themes.Base:
    """Build a warm-dark editorial Gradio theme."""
    gold_hue = gr.themes.Color(
        c50="#fdf8ec", c100="#f5e9c8", c200="#e9d49e",
        c300="#d4b870", c400=GOLD, c500="#b8923a",
        c600="#9a7a30", c700="#7c6226", c800="#5e4a1c",
        c900="#403212", c950="#221a08",
    )
    warm_hue = gr.themes.Color(
        c50="#fdf3ec", c100="#f5ddc8", c200="#e9c09e",
        c300=AMBER, c400="#c07a4a", c500="#a8633a",
        c600="#8a5030", c700="#6c3d26", c800="#4e2a1c",
        c900="#301812", c950="#180c08",
    )
    stone_hue = gr.themes.Color(
        c50=TEXT, c100="#d0c8bc", c200="#b8afa3",
        c300=TEXT_MUTED, c400="#8a8078", c500="#6e655c",
        c600="#524a42", c700="#3a3430", c800=SURFACE_RAISED,
        c900=SURFACE, c950=BG,
    )

    theme = gr.themes.Base(
        primary_hue=gold_hue,
        secondary_hue=warm_hue,
        neutral_hue=stone_hue,
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
        # -- Body --
        body_background_fill=BG,
        body_background_fill_dark=BG,
        body_text_color=TEXT,
        body_text_color_dark=TEXT,
        body_text_color_subdued=TEXT_MUTED,
        body_text_color_subdued_dark=TEXT_MUTED,

        # -- Blocks / panels --
        block_background_fill=SURFACE,
        block_background_fill_dark=SURFACE,
        block_border_color=BORDER,
        block_border_color_dark=BORDER,
        block_border_width="1px",
        block_label_text_color=TEXT_MUTED,
        block_label_text_color_dark=TEXT_MUTED,
        block_title_text_color=TEXT,
        block_title_text_color_dark=TEXT,
        block_shadow="0 2px 16px rgba(0,0,0,0.25)",
        block_shadow_dark="0 2px 16px rgba(0,0,0,0.25)",
        block_radius="10px",

        # -- Inputs --
        input_background_fill=SURFACE_RAISED,
        input_background_fill_dark=SURFACE_RAISED,
        input_border_color=BORDER_STRONG,
        input_border_color_dark=BORDER_STRONG,
        input_placeholder_color=TEXT_DIM,
        input_placeholder_color_dark=TEXT_DIM,

        # -- Primary button (gold) --
        button_primary_background_fill=GOLD,
        button_primary_background_fill_dark=GOLD,
        button_primary_background_fill_hover="#b8923a",
        button_primary_background_fill_hover_dark="#b8923a",
        button_primary_text_color=BG,
        button_primary_text_color_dark=BG,
        button_primary_border_color=GOLD,
        button_primary_border_color_dark=GOLD,

        # -- Secondary button --
        button_secondary_background_fill=SURFACE_RAISED,
        button_secondary_background_fill_dark=SURFACE_RAISED,
        button_secondary_text_color=TEXT,
        button_secondary_text_color_dark=TEXT,
        button_secondary_border_color=BORDER_STRONG,
        button_secondary_border_color_dark=BORDER_STRONG,

        # -- Accents --
        border_color_accent=GOLD,
        border_color_accent_dark=GOLD,
        color_accent=GOLD,
        link_text_color=AMBER,
        link_text_color_dark=AMBER,
        link_text_color_hover=GOLD,
        link_text_color_hover_dark=GOLD,

        # -- Tables --
        table_border_color=BORDER,
        table_border_color_dark=BORDER,
        table_even_background_fill=SURFACE,
        table_even_background_fill_dark=SURFACE,
        table_odd_background_fill=SURFACE_RAISED,
        table_odd_background_fill_dark=SURFACE_RAISED,

        # -- Misc --
        shadow_spread="4px",
        checkbox_background_color=SURFACE_RAISED,
        checkbox_background_color_dark=SURFACE_RAISED,
        checkbox_border_color=BORDER_STRONG,
        checkbox_border_color_dark=BORDER_STRONG,
        checkbox_label_background_fill=SURFACE,
        checkbox_label_background_fill_dark=SURFACE,
    )
    return theme


# ---------------------------------------------------------------------------
# Header / Footer
# ---------------------------------------------------------------------------

HEADER_HTML = """\
<div class="as-header">
  <h1 class="as-header__title">ArtSleuth</h1>
  <p class="as-header__sub">Computational Art Analysis Framework</p>
  <div class="as-header__rule"></div>
</div>
"""

FOOTER_HTML = """\
<div class="as-footer">
  <div class="as-footer__rule"></div>
  <p>ArtSleuth &mdash; Computational Art Analysis &bull; Research use only</p>
  <p class="as-footer__sub">
    Results are probabilistic and should be reviewed by qualified art historians.
    &ensp;&middot;&ensp;
    <a href="https://github.com/ladyFaye1998/ArtSleuth" target="_blank">GitHub</a>
    &ensp;&middot;&ensp;
    <a href="https://huggingface.co/ladyFaye1998/artsleuth-weights" target="_blank">Weights</a>
  </p>
</div>
"""


# ---------------------------------------------------------------------------
# CUSTOM_CSS — the single source of truth for all component styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = r"""
/* ── Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* ── Design tokens as CSS custom properties ────────────────────────── */
:root,
.dark,
.gradio-container {
    --as-bg:            #1a1714;
    --as-surface:       #242019;
    --as-surface-raise: #2e2923;
    --as-border:        rgba(201,168,76,0.12);
    --as-border-strong: rgba(201,168,76,0.25);
    --as-text:          #e8e0d4;
    --as-text-muted:    #a89f94;
    --as-text-dim:      #6e655c;
    --as-gold:          #c9a84c;
    --as-gold-dim:      #8a7434;
    --as-amber:         #d4956a;
    --as-rose:          #c27889;
    --as-green:         #7dab6c;
    --as-red:           #c2555a;
    --as-font:          'Inter', -apple-system, sans-serif;
    --as-font-display:  'Cormorant Garamond', Georgia, serif;
}

/* ── Container ─────────────────────────────────────────────────────── */
.gradio-container {
    font-family: var(--as-font) !important;
    max-width: 1200px;
    margin: 0 auto;
    color: var(--as-text);
}

/* Force dark prose in gr.HTML wrappers */
.prose, .prose-sm,
.prose *, .prose-sm * {
    color: var(--as-text) !important;
}

/* ── Header ────────────────────────────────────────────────────────── */
.as-header {
    text-align: center;
    padding: 2.8rem 1.5rem 2rem;
    position: relative;
}
.as-header__title {
    font-family: var(--as-font-display);
    font-size: 3.2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.08em;
    background: linear-gradient(135deg, #c9a84c 0%, #e6c96e 50%, #c9a84c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.as-header__sub {
    font-family: var(--as-font-display);
    font-size: 1.05rem;
    margin: 0.2rem 0 0;
    color: var(--as-text-muted);
    font-style: italic;
    font-weight: 400;
}
.as-header__rule {
    width: 80px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--as-gold), transparent);
    margin: 1.2rem auto 0;
}

/* ── Tabs ──────────────────────────────────────────────────────────── */
.tab-nav button {
    font-family: var(--as-font) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    color: var(--as-text-muted) !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    background: transparent !important;
}
.tab-nav button:hover {
    color: var(--as-text) !important;
    border-bottom-color: var(--as-gold-dim) !important;
}
.tab-nav button.selected {
    color: var(--as-gold) !important;
    border-bottom: 2px solid var(--as-gold) !important;
}

/* ── Primary button ────────────────────────────────────────────────── */
button.primary {
    background: linear-gradient(135deg, #c9a84c, #b8923a) !important;
    color: var(--as-bg) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.2) !important;
    transition: all 0.3s ease !important;
}
button.primary:hover {
    box-shadow: 0 6px 28px rgba(201,168,76,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Footer ────────────────────────────────────────────────────────── */
.as-footer {
    text-align: center;
    padding: 1.5rem 1rem;
    margin-top: 1.5rem;
    color: var(--as-text-dim);
    font-size: 0.82rem;
}
.as-footer__rule {
    width: 80px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--as-gold-dim), transparent);
    margin: 0 auto 1rem;
}
.as-footer__sub {
    font-size: 0.75rem;
    color: var(--as-text-dim);
    margin-top: 0.3rem;
}
.as-footer a {
    color: var(--as-amber);
    text-decoration: none;
    transition: color 0.2s;
}
.as-footer a:hover {
    color: var(--as-gold);
}

/* ══════════════════════════════════════════════════════════════════════
   Component classes — used by web/components.py and web/app.py
   ══════════════════════════════════════════════════════════════════════ */

/* ── Card ──────────────────────────────────────────────────────────── */
.as-card {
    background: var(--as-surface);
    border: 1px solid var(--as-border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: 0.6rem 0;
    font-family: var(--as-font);
    transition: border-color 0.3s, box-shadow 0.3s;
}
.as-card:hover {
    border-color: var(--as-border-strong);
    box-shadow: 0 0 24px rgba(201,168,76,0.04);
}
.as-card__title {
    margin: 0 0 0.8rem;
    color: var(--as-gold);
    font-family: var(--as-font-display);
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    border-bottom: 1px solid var(--as-border-strong);
    padding-bottom: 0.5rem;
}

/* ── Confidence bar ────────────────────────────────────────────────── */
.as-bar {
    margin: 5px 0;
    font-family: var(--as-font);
}
.as-bar__header {
    display: flex;
    justify-content: space-between;
    font-size: 0.84rem;
    color: var(--as-text);
    font-weight: 500;
    margin-bottom: 3px;
}
.as-bar__pct {
    font-weight: 600;
    color: var(--as-text-muted);
}
.as-bar__track {
    background: var(--as-surface-raise);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.as-bar__fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}

/* Bar colour variants */
.as-bar--gold .as-bar__fill   { background: linear-gradient(90deg, #c9a84c, #d4b870); }
.as-bar--amber .as-bar__fill  { background: linear-gradient(90deg, #d4956a, #e0ad82); }
.as-bar--rose .as-bar__fill   { background: linear-gradient(90deg, #c27889, #d4929f); }
.as-bar--blue .as-bar__fill   { background: linear-gradient(90deg, #7fb3d3, #9ec6df); }
.as-bar--navy .as-bar__fill   { background: linear-gradient(90deg, #5a7da0, #7b9ab8); }
.as-bar--muted .as-bar__fill  { background: linear-gradient(90deg, #6e655c, #8a8078); }

/* ── Badge ─────────────────────────────────────────────────────────── */
.as-badge {
    display: inline-block;
    font-size: 0.65rem;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    vertical-align: middle;
}
.as-badge--gold { background: var(--as-gold); color: var(--as-bg); }
.as-badge--rose { background: var(--as-rose); color: #fff; }
.as-badge--red  { background: var(--as-red);  color: #fff; }
.as-badge--muted { background: var(--as-surface-raise); color: var(--as-text-muted); border: 1px solid var(--as-border-strong); }

/* ── Callout (info / error) ────────────────────────────────────────── */
.as-callout {
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin: 0.4rem 0;
    font-family: var(--as-font);
    font-size: 0.88rem;
    line-height: 1.5;
    border-left: 3px solid;
}
.as-callout--info {
    background: rgba(201,168,76,0.06);
    border-left-color: var(--as-gold-dim);
    color: var(--as-text-muted);
}
.as-callout--error {
    background: rgba(194,85,90,0.08);
    border-left-color: var(--as-red);
    color: var(--as-rose);
}

/* ── Date panel (Estimate Date tab) ────────────────────────────────── */
.as-date {
    text-align: center;
    padding: 1.5rem 1rem;
}
.as-date__year {
    font-family: var(--as-font-display);
    font-size: 3.4rem;
    font-weight: 700;
    color: var(--as-text);
    line-height: 1.1;
}
.as-date__label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--as-gold);
    font-weight: 600;
    margin-top: 0.3rem;
}
.as-date__stats {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin-top: 1.4rem;
    flex-wrap: wrap;
}
.as-date__stat-label {
    font-size: 0.65rem;
    color: var(--as-text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.as-date__stat-value {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--as-text);
    margin-top: 2px;
}
.as-date__stat-value--good { color: var(--as-green); }
.as-date__stat-value--mid  { color: var(--as-amber); }
.as-date__stat-value--bad  { color: var(--as-red); }

.as-date__foot {
    font-size: 0.78rem;
    color: var(--as-text-dim);
    margin-top: 1.2rem;
    font-style: italic;
}

/* ── Similarity panel (Compare tab) ────────────────────────────────── */
.as-sim {
    text-align: center;
    padding: 1.5rem 1rem;
}
.as-sim__value {
    font-family: var(--as-font-display);
    font-size: 3.2rem;
    font-weight: 700;
    color: var(--as-text);
}
.as-sim__label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--as-gold);
    font-weight: 600;
    margin-top: 0.2rem;
}
.as-sim__interp {
    font-size: 0.9rem;
    color: var(--as-text-muted);
    margin-top: 0.8rem;
    max-width: 500px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.5;
}

/* ── Data table ────────────────────────────────────────────────────── */
.as-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--as-font);
    font-size: 0.85rem;
    color: var(--as-text);
}
.as-table thead th {
    padding: 0.65rem 0.8rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--as-gold);
    border-bottom: 1px solid var(--as-border-strong);
}
.as-table thead th:not(:first-child) {
    text-align: center;
}
.as-table tbody td {
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid var(--as-border);
}
.as-table tbody td:not(:first-child) {
    text-align: center;
}
.as-table tbody tr {
    transition: background 0.2s;
}
.as-table tbody tr:hover {
    background: var(--as-surface-raise);
}
.as-table--highlight {
    font-weight: 700;
    color: var(--as-gold);
}

/* ── Gauge (forgery) ───────────────────────────────────────────────── */
.as-gauge {
    text-align: center;
    font-family: var(--as-font);
}
.as-gauge__ring {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
}
.as-gauge__inner {
    width: 112px;
    height: 112px;
    border-radius: 50%;
    background: var(--as-surface);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.15);
}
.as-gauge__score {
    font-family: var(--as-font-display);
    font-size: 2rem;
    font-weight: 700;
    color: var(--as-text);
}
.as-gauge__verdict {
    font-size: 0.62rem;
    color: var(--as-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
.as-gauge__ref {
    font-size: 0.82rem;
    color: var(--as-text-dim);
    margin-top: 0.5rem;
}

/* ── Hand card (workshop) ──────────────────────────────────────────── */
.as-hand {
    border-left: 3px solid var(--as-gold);
    padding: 0.6rem 0.85rem;
    margin: 0.5rem 0;
    background: var(--as-surface-raise);
    border-radius: 0 6px 6px 0;
}
.as-hand__label {
    font-weight: 600;
    color: var(--as-text);
    font-size: 0.95rem;
    margin-bottom: 4px;
}
.as-hand__stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3px 16px;
    font-size: 0.82rem;
    color: var(--as-text-muted);
    margin-top: 6px;
}

/* ── Misc ──────────────────────────────────────────────────────────── */
.as-consensus {
    font-size: 0.9rem;
    color: var(--as-text);
    margin-bottom: 0.6rem;
}
.as-ci {
    font-size: 0.78rem;
    color: var(--as-text-dim);
    margin: -2px 0 8px 4px;
}
.as-note {
    font-size: 0.75rem;
    color: var(--as-text-dim);
    margin-top: 0.6rem;
    font-style: italic;
}

/* ── Benchmark wrapper ─────────────────────────────────────────────── */
.as-bench {
    text-align: center;
    margin-bottom: 1.2rem;
}
.as-bench__title {
    font-family: var(--as-font-display);
    color: var(--as-gold);
    margin: 0 0 0.3rem;
    font-size: 1.4rem;
    font-weight: 600;
}
.as-bench__sub {
    font-size: 0.78rem;
    color: var(--as-text-dim);
}
"""
