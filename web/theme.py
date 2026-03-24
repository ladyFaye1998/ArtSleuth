"""Dark-first Gradio theme for ArtSleuth.

Designed for HuggingFace Spaces where dark mode is the default.
"Gallery at Night" aesthetic — dark surfaces, gold accent lighting,
glass-morphism cards, warm ivory text.
"""
from __future__ import annotations

import gradio as gr

# ── Palette ─────────────────────────────────────────────────────────
BG = "#0c0a09"
SURFACE = "#1c1917"
CARD = "#292524"
BORDER = "rgba(212,168,67,0.18)"
TEXT = "#faf9f6"
TEXT_DIM = "#a8a29e"
GOLD = "#d4a843"
GOLD_BRIGHT = "#f0d060"
ROSE = "#f0768a"
BLUE = "#68b5d5"
SUCCESS = "#4ade80"
WARNING = "#fbbf24"
DANGER = "#f87171"

FONT_DISPLAY = "'Playfair Display', Georgia, serif"
FONT_BODY = "'Inter', -apple-system, sans-serif"

HTML_ELEM_CLASSES: list[str] = ["as-html"]


def artsleuth_theme() -> gr.themes.Base:
    """Build and return the dark ArtSleuth Gradio theme."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#fdf8ef", c100="#faefd3", c200="#f5dfa6",
            c300="#eecb6d", c400=GOLD, c500="#b8922e",
            c600="#967422", c700="#74581b", c800="#5c4517",
            c900="#4d3a16", c950="#2c1f0a",
        ),
        secondary_hue=gr.themes.Color(
            c50="#f0f9ff", c100="#e0f2fe", c200="#bae6fd",
            c300="#7dd3fc", c400=BLUE, c500="#38a3c6",
            c600="#2089a8", c700="#1a6d88", c800="#155a70",
            c900="#12495d", c950="#0a2e3d",
        ),
        neutral_hue=gr.themes.Color(
            c50=TEXT, c100="#f5f5f4", c200="#e7e5e4",
            c300="#d6d3d1", c400=TEXT_DIM, c500="#78716c",
            c600="#57534e", c700="#44403c", c800=CARD,
            c900=SURFACE, c950=BG,
        ),
        font=[gr.themes.GoogleFont("Inter"), "-apple-system", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Fira Code"), "Consolas", "monospace"],
    ).set(
        body_background_fill=BG,
        body_background_fill_dark=BG,
        body_text_color=TEXT,
        body_text_color_dark=TEXT,
        button_primary_background_fill=GOLD,
        button_primary_background_fill_hover="#c4a235",
        button_primary_background_fill_dark=GOLD,
        button_primary_background_fill_hover_dark="#c4a235",
        button_primary_text_color="#0c0a09",
        button_primary_text_color_dark="#0c0a09",
        button_primary_border_color=GOLD,
        button_primary_border_color_dark=GOLD,
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_dark="transparent",
        button_secondary_text_color=TEXT_DIM,
        button_secondary_text_color_dark=TEXT_DIM,
        button_secondary_border_color="rgba(168,162,158,0.3)",
        block_title_text_color=TEXT,
        block_title_text_color_dark=TEXT,
        block_label_text_color=TEXT_DIM,
        block_label_text_color_dark=TEXT_DIM,
        block_background_fill=SURFACE,
        block_background_fill_dark=SURFACE,
        block_border_color=BORDER,
        block_border_color_dark=BORDER,
        block_border_width="1px",
        block_shadow="0 4px 32px rgba(0,0,0,0.4)",
        input_background_fill=CARD,
        input_background_fill_dark=CARD,
        input_border_color="rgba(168,162,158,0.2)",
        input_border_color_dark="rgba(168,162,158,0.2)",
        input_placeholder_color="#78716c",
        border_color_accent=GOLD,
        color_accent=GOLD,
        link_text_color=ROSE,
        link_text_color_hover=GOLD_BRIGHT,
        shadow_spread="6px",
        block_radius="14px",
        table_border_color="rgba(168,162,158,0.15)",
        table_even_background_fill=SURFACE,
        table_odd_background_fill=CARD,
    )
    return theme


# ── HTML fragments ──────────────────────────────────────────────────

HEADER_HTML = f"""
<div class="as-header">
  <div class="as-header__glow"></div>
  <h1>ArtSleuth</h1>
  <p class="as-header__sub">Computational Art Analysis Framework</p>
  <div class="as-header__rule"></div>
  <p class="as-header__tagline">Vision transformers &middot; Brushstroke analysis &middot; Zero-shot attribution</p>
</div>
"""

FOOTER_HTML = """
<div class="as-footer">
  <div class="as-footer__rule"></div>
  <p>ArtSleuth &mdash; Research use only</p>
  <p class="as-footer__disclaimer">
    Results are probabilistic and should be reviewed by qualified art historians.
    <br/>
    <a href="https://github.com/ladyFaye1998/ArtSleuth" target="_blank">GitHub</a>
    &ensp;&bull;&ensp;
    <a href="https://huggingface.co/ladyFaye1998/artsleuth-weights" target="_blank">Weights</a>
  </p>
</div>
"""

# ── Master CSS ──────────────────────────────────────────────────────

CUSTOM_CSS = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400"
    "&family=Inter:wght@300;400;500;600&display=swap');\n"
    """

/* ── Global ─────────────────────────────────────────────── */
.gradio-container {
    font-family: 'Inter', -apple-system, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    background: #0c0a09 !important;
    color: #faf9f6 !important;
}

/* Force dark everywhere — no light-mode leaks */
.dark, :root {
    --body-background-fill: #0c0a09 !important;
    --body-text-color: #faf9f6 !important;
    --block-background-fill: #1c1917 !important;
    --block-border-color: rgba(212,168,67,0.18) !important;
    --input-background-fill: #292524 !important;
}

/* ── Header ─────────────────────────────────────────────── */
.as-header {
    position: relative;
    text-align: center;
    padding: 3rem 2rem 2.2rem;
    border-radius: 18px;
    margin-bottom: 1.2rem;
    background: linear-gradient(160deg, #0c0a09 0%, #1a1510 40%, #1c1410 100%);
    border: 1px solid rgba(212,168,67,0.15);
    overflow: hidden;
}
.as-header__glow {
    position: absolute; inset: 0; pointer-events: none;
    background:
        radial-gradient(600px 300px at 30% 80%, rgba(212,168,67,0.07), transparent 70%),
        radial-gradient(400px 250px at 70% 20%, rgba(240,118,138,0.04), transparent 70%);
}
.as-header h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 3.2rem; font-weight: 700;
    margin: 0; letter-spacing: 0.05em;
    background: linear-gradient(135deg, #d4a843, #f0d060, #d4a843);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
    text-shadow: 0 0 60px rgba(212,168,67,0.2);
}
.as-header__sub {
    font-size: 1rem; font-weight: 300;
    color: #a8a29e; margin: 0.3rem 0 0;
    font-style: italic; position: relative;
}
.as-header__rule {
    width: 80px; height: 1px; margin: 1rem auto 0.8rem;
    background: linear-gradient(90deg, transparent, #d4a843, transparent);
}
.as-header__tagline {
    font-size: 0.75rem; color: #78716c;
    letter-spacing: 0.06em; text-transform: uppercase;
    margin: 0; position: relative;
}

/* ── Tabs ───────────────────────────────────────────────── */
.tab-nav {
    border-bottom: 1px solid rgba(212,168,67,0.12) !important;
}
.tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #78716c !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.8rem 1.2rem !important;
    transition: all 0.3s ease !important;
}
.tab-nav button:hover {
    color: #d4a843 !important;
}
.tab-nav button.selected {
    color: #faf9f6 !important;
    border-bottom-color: #d4a843 !important;
}

/* ── Buttons ────────────────────────────────────────────── */
button.primary {
    background: linear-gradient(135deg, #d4a843 0%, #b8922e 100%) !important;
    color: #0c0a09 !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    box-shadow: 0 0 24px rgba(212,168,67,0.15), 0 4px 12px rgba(0,0,0,0.3) !important;
    transition: all 0.3s ease !important;
}
button.primary:hover {
    box-shadow: 0 0 40px rgba(212,168,67,0.3), 0 6px 20px rgba(0,0,0,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Footer ─────────────────────────────────────────────── */
.as-footer {
    text-align: center; padding: 2rem 1rem; margin-top: 1.5rem;
    color: #78716c; font-size: 0.8rem;
}
.as-footer__rule {
    width: 60px; height: 1px; margin: 0 auto 1.2rem;
    background: linear-gradient(90deg, transparent, rgba(212,168,67,0.3), transparent);
}
.as-footer p { margin: 0.3rem 0; }
.as-footer__disclaimer { font-size: 0.72rem; color: #57534e; }
.as-footer a { color: #f0768a; text-decoration: none; transition: color 0.2s; }
.as-footer a:hover { color: #d4a843; }

/* ── gr.HTML result host ────────────────────────────────── */
.as-html {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.as-html > .prose,
.as-html > div > .prose {
    color: #faf9f6 !important;
    max-width: none !important;
}
.as-html .prose * {
    color: inherit !important;
}

/* ── Glass card used by components.py & app.py ──────────── */
.as-card {
    background: linear-gradient(180deg, rgba(28,25,23,0.92) 0%, rgba(28,25,23,0.98) 100%) !important;
    border: 1px solid rgba(212,168,67,0.12) !important;
    border-radius: 14px !important;
    padding: 1.4rem 1.5rem !important;
    margin: 0.6rem 0 !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    color: #faf9f6 !important;
}
.as-card h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #d4a843 !important;
    margin: 0 0 1rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid rgba(212,168,67,0.15) !important;
    letter-spacing: 0.03em !important;
}

/* ── Bars ───────────────────────────────────────────────── */
.as-bar-wrap { margin: 8px 0; }
.as-bar-top {
    display: flex; justify-content: space-between;
    font-size: 0.84rem; color: #e7e5e4; margin-bottom: 4px;
}
.as-bar-top .as-pct { color: #d4a843; font-weight: 600; font-variant-numeric: tabular-nums; }
.as-bar-track {
    height: 6px; border-radius: 6px;
    background: rgba(255,255,255,0.06);
    overflow: hidden;
}
.as-bar-fill {
    height: 100%; border-radius: 6px;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}

/* ── Info / Error callouts ──────────────────────────────── */
.as-msg {
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    font-size: 0.88rem !important;
    line-height: 1.5 !important;
    margin: 0.4rem 0 !important;
}
.as-msg--info {
    background: rgba(104,181,213,0.10) !important;
    border: 1px solid rgba(104,181,213,0.25) !important;
    color: #bae6fd !important;
}
.as-msg--error {
    background: rgba(248,113,113,0.10) !important;
    border: 1px solid rgba(248,113,113,0.25) !important;
    color: #fca5a5 !important;
}

/* ── Date panel ─────────────────────────────────────────── */
.as-date {
    text-align: center; padding: 1.5rem;
}
.as-date__year {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 3.4rem; font-weight: 700;
    color: #faf9f6; line-height: 1;
}
.as-date__label {
    font-size: 0.72rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #d4a843;
    font-weight: 600; margin-top: 0.3rem;
}
.as-date__stats {
    display: flex; justify-content: center; gap: 2.5rem;
    margin-top: 1.4rem; flex-wrap: wrap;
}
.as-date__stat-label {
    font-size: 0.65rem; color: #78716c;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.as-date__stat-value {
    font-size: 1.15rem; font-weight: 600;
    color: #e7e5e4; margin-top: 2px;
}
.as-date__foot {
    font-size: 0.76rem; color: #78716c;
    margin-top: 1.2rem; font-style: italic;
}
.as-score-good { color: #4ade80 !important; }
.as-score-mid  { color: #fbbf24 !important; }
.as-score-bad  { color: #f87171 !important; }

/* ── Badge ──────────────────────────────────────────────── */
.as-badge {
    display: inline-block; font-size: 0.65rem;
    padding: 3px 10px; border-radius: 20px;
    font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Benchmark table ────────────────────────────────────── */
table { font-size: 0.85rem !important; }
table th {
    background: #292524 !important;
    color: #d4a843 !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border-bottom: 1px solid rgba(212,168,67,0.2) !important;
}
table td {
    color: #e7e5e4 !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
}
table tr:hover td { background: rgba(212,168,67,0.04) !important; }
"""
)
