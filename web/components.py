"""Reusable UI component builders for the ArtSleuth Gradio app.

Every ``build_*`` / ``format_*`` function returns a self-contained HTML
string that uses CSS classes defined in ``web.theme.CUSTOM_CSS``.
No inline ``style="color:…"`` attributes — all colour and layout comes
from ``.as-*`` classes so Gradio's ``.prose`` wrapper never fights the
design.
"""
from __future__ import annotations

import html as _html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from artsleuth.core.attribution import AttributionReport
    from artsleuth.core.forgery import ForgeryReport
    from artsleuth.core.style import StyleReport
    from artsleuth.core.temporal import TemporalPrediction
    from artsleuth.core.workshop import WorkshopReport

# Bar-colour rotation (CSS class suffixes defined in CUSTOM_CSS)
_BAR_VARIANTS = ("gold", "amber", "rose", "blue", "navy", "muted")


def _esc(text: str) -> str:
    """HTML-escape user-supplied text."""
    return _html.escape(str(text))


def _pct(value: float) -> str:
    """Format a 0-1 float as a percentage string."""
    return f"{value * 100:.1f}%"


# ---------------------------------------------------------------------------
# Atomic elements
# ---------------------------------------------------------------------------


def _bar_html(
    label: str,
    value: float,
    variant: str = "gold",
) -> str:
    """Horizontal confidence bar using ``.as-bar`` classes."""
    pct = min(max(value * 100, 0), 100)
    return (
        f'<div class="as-bar as-bar--{variant}">'
        f'  <div class="as-bar__header">'
        f"    <span>{_esc(label)}</span>"
        f'    <span class="as-bar__pct">{pct:.1f}%</span>'
        f"  </div>"
        f'  <div class="as-bar__track">'
        f'    <div class="as-bar__fill" style="width:{pct:.1f}%"></div>'
        f"  </div>"
        f"</div>"
    )


def _card(title: str, body: str) -> str:
    """Wrap *body* in a titled ``.as-card``."""
    return (
        f'<div class="as-card">'
        f'  <h3 class="as-card__title">{_esc(title)}</h3>'
        f"  {body}"
        f"</div>"
    )


def _badge(text: str, variant: str = "gold") -> str:
    """Inline badge — ``variant`` is gold | rose | red | muted."""
    return (
        f'<span class="as-badge as-badge--{variant}">'
        f"{_esc(text)}</span>"
    )


# ---------------------------------------------------------------------------
# Public API — layout builders
# ---------------------------------------------------------------------------


def build_header() -> str:
    """Return HTML for the ArtSleuth app header banner."""
    from web.theme import HEADER_HTML

    return HEADER_HTML


def build_footer() -> str:
    """Return HTML for the ArtSleuth footer."""
    from web.theme import FOOTER_HTML

    return FOOTER_HTML


# ---------------------------------------------------------------------------
# Public API — report formatters
# ---------------------------------------------------------------------------


def format_style_report(style_report: StyleReport) -> str:
    """Render a :class:`StyleReport` as rich HTML with coloured bars."""
    axes = [
        ("Period", style_report.period, "gold"),
        ("School", style_report.school, "amber"),
        ("Genre", style_report.technique, "rose"),
    ]

    parts: list[str] = []
    for axis_name, pred, variant in axes:
        bars = "".join(
            _bar_html(label, conf, variant=variant)
            for label, conf in pred.top_k
        )
        parts.append(_card(
            f"{axis_name}: {_esc(pred.label)} ({_pct(pred.confidence)})",
            bars,
        ))

    return '<div class="as-style-wrap">' + "".join(parts) + "</div>"


def format_attribution_report(
    attribution_report: AttributionReport,
) -> str:
    """Render an :class:`AttributionReport` as HTML."""
    rows: list[str] = []
    for rank, cand in enumerate(
        attribution_report.candidates, start=1,
    ):
        lo, hi = cand.confidence_interval
        ci_text = f"95% CI: [{lo:.2f}, {hi:.2f}]"
        features = ", ".join(cand.supporting_features) or "\u2014"
        variant = "gold" if rank == 1 else "amber"
        bar = _bar_html(
            f"#{rank} {cand.artist}", cand.score, variant=variant,
        )
        rows.append(
            f"{bar}"
            f'<div class="as-ci">'
            f"  {_esc(ci_text)} &middot; {_esc(features)}"
            f"</div>"
        )

    header_extra = ""
    if attribution_report.multi_hand_flag:
        header_extra = f" {_badge('MULTI-HAND', 'rose')}"

    body = "".join(rows)
    consensus = (
        f'<div class="as-consensus">'
        f"  <strong>Consensus:</strong> "
        f"  {_esc(attribution_report.consensus_artist)}"
        f"  ({_pct(attribution_report.consensus_confidence)})"
        f"  {header_extra}"
        f"</div>"
    )

    return _card("Attribution Candidates", consensus + body)


def format_forgery_gauge(forgery_report: ForgeryReport) -> str:
    """Render a circular gauge for the forgery anomaly score."""
    score = forgery_report.anomaly_score
    score_pct = score * 100

    if score < 0.4:
        ring_color = "var(--as-green)"
        verdict = "LOW RISK"
    elif score < 0.7:
        ring_color = "var(--as-amber)"
        verdict = "MODERATE"
    else:
        ring_color = "var(--as-red)"
        verdict = "HIGH RISK"

    flag_badge = ""
    if forgery_report.is_flagged:
        flag_badge = _badge("FLAGGED", "red")

    indicator_rows = ""
    for ind in forgery_report.indicators:
        if ind.z_score > 3.0:
            z_cls = "as-date__stat-value--bad"
        elif ind.z_score > 2.0:
            z_cls = "as-date__stat-value--mid"
        else:
            z_cls = "as-date__stat-value--good"
        indicator_rows += (
            f"<tr>"
            f"  <td>{_esc(ind.feature_name)}</td>"
            f'  <td class="{z_cls}" style="font-weight:600">'
            f"    z\u202f=\u202f{ind.z_score:.1f}</td>"
            f"  <td>{_esc(ind.description)}</td>"
            f"</tr>"
        )

    gauge_html = (
        f'<div class="as-gauge">'
        f'  <div class="as-gauge__ring" style="background:conic-gradient('
        f"    {ring_color} 0deg {score_pct * 3.6:.1f}deg,"
        f"    var(--as-surface-raise) {score_pct * 3.6:.1f}deg 360deg"
        f'  )">'
        f'    <div class="as-gauge__inner">'
        f'      <span class="as-gauge__score">{score:.2f}</span>'
        f'      <span class="as-gauge__verdict">{verdict}</span>'
        f"    </div>"
        f"  </div>"
        f'  <div style="margin-top:0.6rem">{flag_badge}</div>'
        f'  <div class="as-gauge__ref">'
        f"    Reference: {_esc(forgery_report.reference_artist)}"
        f"  </div>"
        f"</div>"
    )

    table_html = ""
    if forgery_report.indicators:
        table_html = (
            f'<table class="as-table" style="margin-top:0.8rem">'
            f"<thead><tr>"
            f"  <th>Feature</th><th>Z-Score</th><th>Detail</th>"
            f"</tr></thead><tbody>"
            f"  {indicator_rows}"
            f"</tbody></table>"
        )

    return _card("Forgery Screening", gauge_html + table_html)


def format_workshop_report(
    workshop_report: WorkshopReport,
) -> str:
    """Render a :class:`WorkshopReport` as HTML."""
    workshop_badge = ""
    if workshop_report.is_workshop:
        workshop_badge = _badge("WORKSHOP PRODUCTION", "gold")

    summary = (
        f'<div class="as-consensus">'
        f"  <strong>Detected hands:</strong> "
        f"  {workshop_report.num_hands} {workshop_badge}"
        f"</div>"
    )

    hand_variants = ("gold", "rose", "amber", "blue", "navy", "muted")

    cards: list[str] = []
    for idx, hand in enumerate(workshop_report.assignments):
        variant = hand_variants[idx % len(hand_variants)]
        readable_label = hand.label.replace("_", " ").title()
        extent_bar = _bar_html(
            "Spatial extent", hand.spatial_extent, variant=variant,
        )
        conf_bar = _bar_html(
            "Confidence", hand.confidence, variant=variant,
        )

        cards.append(
            f'<div class="as-hand">'
            f'  <div class="as-hand__label">{_esc(readable_label)}</div>'
            f"  {extent_bar}{conf_bar}"
            f'  <div class="as-hand__stats">'
            f"    <span>Patches: {hand.patch_count}</span>"
            f"    <span>Coherence: {hand.mean_coherence:.3f}</span>"
            f"    <span>Energy: {hand.mean_energy:.3f}</span>"
            f"    <span>Hand ID: {hand.hand_id}</span>"
            f"  </div>"
            f"</div>"
        )

    return _card(
        "Workshop Decomposition",
        summary + "".join(cards),
    )


def format_temporal_prediction(
    temporal_prediction: TemporalPrediction,
) -> str:
    """Render a :class:`TemporalPrediction` as HTML."""
    year = temporal_prediction.estimated_year
    lo, hi = temporal_prediction.confidence_band
    score = temporal_prediction.temporal_score
    drift = temporal_prediction.drift_rate

    year_display = str(round(year))
    lo_display = str(round(lo))
    hi_display = str(round(hi))

    score_cls = (
        "as-date__stat-value--good" if score > 0.7
        else "as-date__stat-value--mid" if score > 0.4
        else "as-date__stat-value--bad"
    )

    return (
        f'<div class="as-card">'
        f'  <h3 class="as-card__title">Temporal Analysis</h3>'
        f'  <div class="as-date">'
        f'    <div class="as-date__year">c.\u2009{year_display}</div>'
        f'    <div class="as-date__label">Estimated Date</div>'
        f'    <div class="as-date__stats">'
        f"      <div>"
        f'        <div class="as-date__stat-label">95% Band</div>'
        f'        <div class="as-date__stat-value">'
        f"          {lo_display}\u2009\u2013\u2009{hi_display}</div>"
        f"      </div>"
        f"      <div>"
        f'        <div class="as-date__stat-label">Plausibility</div>'
        f'        <div class="as-date__stat-value {score_cls}">'
        f"          {score:.2f}</div>"
        f"      </div>"
        f"      <div>"
        f'        <div class="as-date__stat-label">Drift / Decade</div>'
        f'        <div class="as-date__stat-value">{drift:.3f}</div>'
        f"      </div>"
        f"    </div>"
        f"  </div>"
        f"</div>"
    )
