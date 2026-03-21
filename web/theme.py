"""Custom Gradio theme for ArtSleuth.

Defines a cohesive visual identity based on the ArtSleuth design
system: navy primary, soft-blue secondary, rose accent, gold tertiary,
and cream neutrals — all set in the *Lora* serif typeface.
"""
from __future__ import annotations

import gradio as gr

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

NAVY = "#1A2E48"
SOFT_BLUE = "#9DC0D8"
ROSE = "#D4899A"
GOLD = "#d4af37"
CREAM = "#F5F0EB"
TEXT_LIGHT = "#F0F0F0"
WHITE = "#FFFFFF"


def artsleuth_theme() -> gr.themes.Base:
    """Build and return the ArtSleuth Gradio theme."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e8edf3",
            c100="#c5d2e2",
            c200="#9db4cf",
            c300="#7596bc",
            c400="#4e78a9",
            c500=NAVY,
            c600="#17283f",
            c700="#132136",
            c800="#0f1b2d",
            c900="#0b1424",
            c950="#070e1b",
        ),
        secondary_hue=gr.themes.Color(
            c50="#f0f6fa",
            c100="#daeaf3",
            c200="#c4deec",
            c300=SOFT_BLUE,
            c400="#7eb0cb",
            c500="#5fa0be",
            c600="#4a8fae",
            c700="#3a7a98",
            c800="#2e6682",
            c900="#22516b",
            c950="#163d55",
        ),
        neutral_hue=gr.themes.Color(
            c50=CREAM,
            c100="#ede6df",
            c200="#e0d6cc",
            c300="#c8bdb2",
            c400="#b0a498",
            c500="#988b7e",
            c600="#807264",
            c700="#68594a",
            c800="#504030",
            c900="#382816",
            c950="#201000",
        ),
        font=[
            gr.themes.GoogleFont("Lora"),
            "Georgia",
            "serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("Fira Code"),
            "Consolas",
            "monospace",
        ],
    ).set(
        # Global
        body_background_fill=CREAM,
        body_text_color=NAVY,
        # Buttons — primary
        button_primary_background_fill=NAVY,
        button_primary_background_fill_hover="#243a58",
        button_primary_text_color=CREAM,
        button_primary_border_color=NAVY,
        # Buttons — secondary
        button_secondary_background_fill=SOFT_BLUE,
        button_secondary_background_fill_hover="#89b3cb",
        button_secondary_text_color=NAVY,
        button_secondary_border_color=SOFT_BLUE,
        # Blocks
        block_title_text_color=NAVY,
        block_label_text_color=NAVY,
        block_background_fill=WHITE,
        block_border_color=SOFT_BLUE,
        block_border_width="1px",
        block_shadow="0 1px 4px rgba(26,46,72,0.08)",
        # Inputs
        input_background_fill=WHITE,
        input_border_color=SOFT_BLUE,
        input_placeholder_color="#988b7e",
        # Checkboxes / radios
        checkbox_label_background_fill=CREAM,
        checkbox_background_color=WHITE,
        checkbox_border_color=SOFT_BLUE,
        # Accordions & tabs
        border_color_accent=GOLD,
        color_accent=GOLD,
        # Links
        link_text_color=ROSE,
        link_text_color_hover=GOLD,
        # Shadows & radius
        shadow_spread="4px",
        block_radius="6px",
        # Table
        table_border_color=SOFT_BLUE,
        table_even_background_fill=CREAM,
        table_odd_background_fill=WHITE,
    )
    return theme


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url(
    'https://fonts.googleapis.com/css2?family=Lora:'
    'ital,wght@0,400;0,600;0,700;1,400&display=swap'
);

.gradio-container {
    font-family: 'Lora', Georgia, serif !important;
    max-width: 1200px;
    margin: 0 auto;
}

/* ---- Header ---- */
.artsleuth-header {
    background: linear-gradient(135deg, #1A2E48 0%, #243a58 100%);
    color: #F5F0EB;
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
.artsleuth-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.3rem;
    letter-spacing: 0.04em;
    color: #d4af37;
}
.artsleuth-header p {
    font-size: 1.05rem;
    margin: 0;
    color: #F0F0F0;
    font-style: italic;
}

/* ---- Forgery gauge ---- */
.artsleuth-gauge {
    width: 160px;
    height: 160px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto;
    background: conic-gradient(
        #4caf50 0deg 120deg,
        #ff9800 120deg 240deg,
        #f44336 240deg 360deg
    );
    position: relative;
}
.artsleuth-gauge .gauge-inner {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}
.artsleuth-gauge .gauge-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1A2E48;
}
.artsleuth-gauge .gauge-label {
    font-size: 0.75rem;
    color: #988b7e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ---- Tables ---- */
table.artsleuth-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
}
table.artsleuth-table th {
    background: #1A2E48;
    color: #F5F0EB;
    padding: 0.55rem 0.75rem;
    text-align: left;
    font-weight: 600;
}
table.artsleuth-table td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #9DC0D8;
}
table.artsleuth-table tr:nth-child(even) {
    background: #F5F0EB;
}
table.artsleuth-table tr:nth-child(odd) {
    background: #FFFFFF;
}

/* ---- Tabs ---- */
.tab-nav button {
    font-family: 'Lora', Georgia, serif !important;
    font-weight: 600;
    color: #1A2E48;
    border-bottom: 2px solid transparent;
    transition: all 0.2s ease;
}
.tab-nav button:hover {
    color: #9DC0D8;
    border-bottom-color: #9DC0D8;
}
.tab-nav button.selected {
    color: #1A2E48 !important;
    border-bottom: 3px solid #1A2E48 !important;
}

/* ---- Footer ---- */
.artsleuth-footer {
    text-align: center;
    padding: 1.2rem 1rem;
    margin-top: 1.5rem;
    border-top: 1px solid #9DC0D8;
    color: #68594a;
    font-size: 0.85rem;
}
.artsleuth-footer a {
    color: #D4899A;
    text-decoration: none;
}
.artsleuth-footer a:hover {
    color: #d4af37;
    text-decoration: underline;
}
"""
