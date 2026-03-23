"""Reusable UI component builders for the ArtSleuth Gradio app.

Every ``build_*`` / ``format_*`` function returns a self-contained HTML
string that uses inline styles drawn from the ArtSleuth palette so it
renders correctly inside a ``gr.HTML`` component.
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

# ---------------------------------------------------------------------------
# Palette constants (kept in sync with theme.py)
# ---------------------------------------------------------------------------

_NAVY = "#1A2E48"
_BLUE = "#9DC0D8"
_ROSE = "#D4899A"
_GOLD = "#d4af37"
_CREAM = "#F5F0EB"
_TEXT_LIGHT = "#F0F0F0"
_WHITE = "#FFFFFF"
_MUTED = "#68594a"
_FONT = "'Lora', Georgia, serif"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """HTML-escape user-supplied text."""
    return _html.escape(str(text))


def _pct(value: float) -> str:
    """Format a 0-1 float as a percentage string."""
    return f"{value * 100:.1f}%"


def _bar_html(
    label: str,
    value: float,
    bar_color: str = _BLUE,
    max_width: str = "100%",
) -> str:
    """Render a single labelled horizontal confidence bar."""
    pct = min(max(value * 100, 0), 100)
    return (
        f'<div style="margin:4px 0;font-family:{_FONT};">'
        f'  <div style="display:flex;justify-content:space-between;'
        f'    font-size:0.88rem;color:{_NAVY};">'
        f"    <span>{_esc(label)}</span>"
        f"    <span>{pct:.1f}%</span>"
        f"  </div>"
        f'  <div style="background:{_CREAM};border-radius:4px;'
        f'    height:10px;width:{max_width};overflow:hidden;">'
        f'    <div style="width:{pct:.1f}%;height:100%;'
        f"      background:{bar_color};border-radius:4px;"
        f'      transition:width 0.4s ease;"></div>'
        f"  </div>"
        f"</div>"
    )


def _section(title: str, body: str) -> str:
    """Wrap *body* in a titled card section."""
    return (
        f'<div style="background:{_WHITE};border:1px solid {_BLUE};'
        f'  border-radius:6px;padding:1rem;margin:0.75rem 0;'
        f'  font-family:{_FONT};">'
        f'  <h3 style="margin:0 0 0.6rem;color:{_NAVY};'
        f'    font-size:1.05rem;border-bottom:2px solid {_GOLD};'
        f'    padding-bottom:0.3rem;">{_esc(title)}</h3>'
        f"  {body}"
        f"</div>"
    )


def _badge(text: str, bg: str = _ROSE, fg: str = _WHITE) -> str:
    """Small inline badge."""
    return (
        f'<span style="display:inline-block;background:{bg};'
        f"  color:{fg};font-size:0.75rem;padding:2px 8px;"
        f'  border-radius:10px;font-weight:600;">'
        f"  {_esc(text)}"
        f"</span>"
    )


# ---------------------------------------------------------------------------
# Public API — layout builders
# ---------------------------------------------------------------------------


def build_header() -> str:
    """Return HTML for the ArtSleuth app header banner."""
    return (
        '<div class="artsleuth-header">'
        "  <h1>ArtSleuth</h1>"
        "  <p>AI Art Forensics &amp; Analysis Framework</p>"
        "</div>"
    )


def build_footer() -> str:
    """Return HTML for the ArtSleuth footer."""
    return (
        '<div class="artsleuth-footer">'
        "  ArtSleuth &mdash; AI Art Forensics &amp; Analysis"
        "  Framework &bull; Research use only<br>"
        '  <span style="font-size:0.78rem;color:#988b7e;">'
        "    Results are probabilistic and should be reviewed"
        "    by qualified art historians."
        "  </span>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Public API — report formatters
# ---------------------------------------------------------------------------


def format_style_report(style_report: StyleReport) -> str:
    """Render a :class:`StyleReport` as rich HTML with coloured bars.

    Parameters
    ----------
    style_report:
        Output of :meth:`StyleClassifier.classify`.

    Returns
    -------
    str
        Self-contained HTML fragment.
    """
    axes = [
        ("Period", style_report.period, _NAVY),
        ("School", style_report.school, _BLUE),
        ("Genre", style_report.technique, _ROSE),
    ]

    parts: list[str] = []
    for axis_name, pred, color in axes:
        bars = "".join(
            _bar_html(label, conf, bar_color=color)
            for label, conf in pred.top_k
        )
        parts.append(_section(
            f"{axis_name}: {_esc(pred.label)} "
            f"({_pct(pred.confidence)})",
            bars,
        ))

    return (
        f'<div style="font-family:{_FONT};">'
        + "".join(parts)
        + "</div>"
    )


def format_attribution_report(
    attribution_report: AttributionReport,
) -> str:
    """Render an :class:`AttributionReport` as HTML.

    Shows ranked candidates with confidence bars, credible-interval
    annotations, and a multi-hand warning badge when applicable.

    Parameters
    ----------
    attribution_report:
        Output of :meth:`AttributionAnalyzer.attribute`.

    Returns
    -------
    str
        Self-contained HTML fragment.
    """
    rows: list[str] = []
    for rank, cand in enumerate(
        attribution_report.candidates, start=1
    ):
        lo, hi = cand.confidence_interval
        ci_text = f"95% CI: [{lo:.2f}, {hi:.2f}]"
        features = ", ".join(cand.supporting_features) or "—"
        bar = _bar_html(
            f"#{rank} {cand.artist}",
            cand.score,
            bar_color=_NAVY if rank == 1 else _BLUE,
        )
        rows.append(
            f"{bar}"
            f'<div style="font-size:0.8rem;color:{_MUTED};'
            f'  margin:-2px 0 8px 4px;">'
            f"  {_esc(ci_text)} &middot; {_esc(features)}"
            f"</div>"
        )

    header_extra = ""
    if attribution_report.multi_hand_flag:
        header_extra = (
            f" {_badge('MULTI-HAND', bg=_GOLD, fg=_NAVY)}"
        )

    body = "".join(rows)
    consensus = (
        f'<div style="font-size:0.9rem;color:{_NAVY};'
        f'  margin-bottom:0.6rem;">'
        f"  <strong>Consensus:</strong> "
        f"  {_esc(attribution_report.consensus_artist)}"
        f"  ({_pct(attribution_report.consensus_confidence)})"
        f"  {header_extra}"
        f"</div>"
    )

    return _section("Attribution Candidates", consensus + body)


def format_forgery_gauge(forgery_report: ForgeryReport) -> str:
    """Render a circular gauge for the forgery anomaly score.

    The ring colour shifts green -> yellow -> red based on score.

    Parameters
    ----------
    forgery_report:
        Output of :meth:`ForgeryDetector.detect`.

    Returns
    -------
    str
        Self-contained HTML fragment.
    """
    score = forgery_report.anomaly_score
    score_pct = score * 100

    if score < 0.4:
        ring_color = "#4caf50"
        verdict = "LOW RISK"
    elif score < 0.7:
        ring_color = "#ff9800"
        verdict = "MODERATE"
    else:
        ring_color = "#f44336"
        verdict = "HIGH RISK"

    flag_badge = ""
    if forgery_report.is_flagged:
        flag_badge = _badge("FLAGGED", bg="#f44336")

    indicator_rows = ""
    for ind in forgery_report.indicators:
        z_color = (
            "#f44336" if ind.z_score > 3.0
            else "#ff9800" if ind.z_score > 2.0
            else "#4caf50"
        )
        indicator_rows += (
            f'<tr style="font-size:0.85rem;">'
            f'  <td style="padding:4px 8px;border-bottom:'
            f'    1px solid {_BLUE};">'
            f"    {_esc(ind.feature_name)}</td>"
            f'  <td style="padding:4px 8px;border-bottom:'
            f'    1px solid {_BLUE};color:{z_color};'
            f'    font-weight:600;">'
            f"    z\u202f=\u202f{ind.z_score:.1f}</td>"
            f'  <td style="padding:4px 8px;border-bottom:'
            f'    1px solid {_BLUE};color:{_MUTED};'
            f'    font-size:0.82rem;">'
            f"    {_esc(ind.description)}</td>"
            f"</tr>"
        )

    gauge_html = (
        f'<div style="text-align:center;font-family:{_FONT};">'
        # Outer ring
        f'  <div style="width:160px;height:160px;'
        f"    border-radius:50%;margin:0 auto;"
        f"    background:conic-gradient("
        f"      {ring_color} 0deg {score_pct * 3.6:.1f}deg,"
        f"      {_CREAM} {score_pct * 3.6:.1f}deg 360deg"
        f'    );">'
        # Inner disc
        f'    <div style="width:120px;height:120px;'
        f"      border-radius:50%;background:{_WHITE};"
        f"      position:relative;top:20px;left:20px;"
        f"      display:flex;align-items:center;"
        f'      justify-content:center;flex-direction:column;">'
        f'      <span style="font-size:1.8rem;font-weight:700;'
        f'        color:{_NAVY};">{score:.2f}</span>'
        f'      <span style="font-size:0.7rem;color:{_MUTED};'
        f'        text-transform:uppercase;letter-spacing:0.06em;">'
        f"        {verdict}</span>"
        f"    </div>"
        f"  </div>"
        # Flag
        f'  <div style="margin-top:0.5rem;">{flag_badge}</div>'
        f'  <div style="font-size:0.82rem;color:{_MUTED};'
        f'    margin-top:0.3rem;">'
        f"    Reference: {_esc(forgery_report.reference_artist)}"
        f"  </div>"
        f"</div>"
    )

    table_html = ""
    if forgery_report.indicators:
        table_html = (
            f'<table style="width:100%;border-collapse:collapse;'
            f'  margin-top:0.8rem;font-family:{_FONT};">'
            f'  <tr style="background:{_NAVY};color:{_CREAM};">'
            f'    <th style="padding:6px 8px;text-align:left;">'
            f"      Feature</th>"
            f'    <th style="padding:6px 8px;text-align:left;">'
            f"      Z-Score</th>"
            f'    <th style="padding:6px 8px;text-align:left;">'
            f"      Detail</th>"
            f"  </tr>"
            f"  {indicator_rows}"
            f"</table>"
        )

    return _section(
        "Forgery Screening",
        gauge_html + table_html,
    )


def format_workshop_report(
    workshop_report: WorkshopReport,
) -> str:
    """Render a :class:`WorkshopReport` as HTML.

    Parameters
    ----------
    workshop_report:
        Output of :meth:`WorkshopDecomposition.decompose`.

    Returns
    -------
    str
        Self-contained HTML fragment with per-hand statistics.
    """
    workshop_badge = ""
    if workshop_report.is_workshop:
        workshop_badge = _badge(
            "WORKSHOP PRODUCTION", bg=_GOLD, fg=_NAVY,
        )

    summary = (
        f'<div style="font-size:0.9rem;color:{_NAVY};'
        f'  margin-bottom:0.6rem;font-family:{_FONT};">'
        f"  <strong>Detected hands:</strong> "
        f"  {workshop_report.num_hands} {workshop_badge}"
        f"</div>"
    )

    hand_colors = [_NAVY, _ROSE, _GOLD, _BLUE, "#8e6bb0", "#5fa0be"]

    cards: list[str] = []
    for idx, hand in enumerate(workshop_report.assignments):
        color = hand_colors[idx % len(hand_colors)]
        readable_label = hand.label.replace("_", " ").title()
        extent_bar = _bar_html(
            "Spatial extent", hand.spatial_extent, bar_color=color,
        )
        conf_bar = _bar_html(
            "Confidence", hand.confidence, bar_color=color,
        )

        stats = (
            f'<div style="display:grid;'
            f"  grid-template-columns:1fr 1fr;gap:4px 16px;"
            f'  font-size:0.84rem;color:{_MUTED};margin-top:6px;">'
            f"  <span>Patches: {hand.patch_count}</span>"
            f"  <span>Coherence: {hand.mean_coherence:.3f}</span>"
            f"  <span>Energy: {hand.mean_energy:.3f}</span>"
            f"  <span>Hand ID: {hand.hand_id}</span>"
            f"</div>"
        )

        cards.append(
            f'<div style="border-left:4px solid {color};'
            f"  padding:0.5rem 0.75rem;margin:0.5rem 0;"
            f'  background:{_CREAM};border-radius:0 4px 4px 0;">'
            f'  <div style="font-weight:600;color:{_NAVY};'
            f'    font-size:0.95rem;">'
            f"    {_esc(readable_label)}"
            f"  </div>"
            f"  {extent_bar}{conf_bar}{stats}"
            f"</div>"
        )

    return _section(
        "Workshop Decomposition",
        summary + "".join(cards),
    )


def format_temporal_prediction(
    temporal_prediction: TemporalPrediction,
) -> str:
    """Render a :class:`TemporalPrediction` as HTML with a timeline.

    Parameters
    ----------
    temporal_prediction:
        Output of :meth:`TemporalStyleModel.predict`.

    Returns
    -------
    str
        Self-contained HTML fragment.
    """
    year = temporal_prediction.estimated_year
    lo, hi = temporal_prediction.confidence_band
    score = temporal_prediction.temporal_score
    drift = temporal_prediction.drift_rate

    year_display = (
        str(round(year)) if year == year else str(round(year))
    )
    lo_display = str(round(lo))
    hi_display = str(round(hi))

    score_color = (
        "#4caf50" if score > 0.7
        else "#ff9800" if score > 0.4
        else "#f44336"
    )

    span = max(hi - lo, 1.0)
    marker_pct = min(
        max((year - lo) / span * 100, 2), 98,
    )

    timeline_html = (
        f'<div style="position:relative;height:40px;'
        f"  margin:1rem 0 0.5rem;background:linear-gradient("
        f"  to right, {_BLUE}33, {_NAVY}33);"
        f'  border-radius:4px;font-family:{_FONT};">'
        # Marker
        f'  <div style="position:absolute;'
        f"    left:{marker_pct:.1f}%;top:-6px;"
        f"    transform:translateX(-50%);"
        f'    text-align:center;">'
        f'    <div style="width:3px;height:52px;'
        f'      background:{_GOLD};margin:0 auto;"></div>'
        f'    <span style="font-size:0.82rem;font-weight:700;'
        f'      color:{_NAVY};">{year_display}</span>'
        f"  </div>"
        # Left label
        f'  <span style="position:absolute;left:4px;bottom:-18px;'
        f'    font-size:0.72rem;color:{_MUTED};">'
        f"    {lo_display}</span>"
        # Right label
        f'  <span style="position:absolute;right:4px;bottom:-18px;'
        f'    font-size:0.72rem;color:{_MUTED};">'
        f"    {hi_display}</span>"
        f"</div>"
    )

    stats_html = (
        f'<div style="display:grid;'
        f"  grid-template-columns:1fr 1fr;gap:8px;"
        f"  margin-top:1.5rem;font-size:0.88rem;"
        f'  font-family:{_FONT};">'
        # Estimated year
        f'  <div style="background:{_CREAM};padding:0.6rem;'
        f'    border-radius:4px;text-align:center;">'
        f'    <div style="font-size:0.72rem;color:{_MUTED};'
        f'      text-transform:uppercase;">Estimated Year</div>'
        f'    <div style="font-size:1.3rem;font-weight:700;'
        f'      color:{_NAVY};">c.\u2009{year_display}</div>'
        f"  </div>"
        # Temporal plausibility
        f'  <div style="background:{_CREAM};padding:0.6rem;'
        f'    border-radius:4px;text-align:center;">'
        f'    <div style="font-size:0.72rem;color:{_MUTED};'
        f'      text-transform:uppercase;">Plausibility</div>'
        f'    <div style="font-size:1.3rem;font-weight:700;'
        f'      color:{score_color};">{score:.2f}</div>'
        f"  </div>"
        # Confidence band
        f'  <div style="background:{_CREAM};padding:0.6rem;'
        f'    border-radius:4px;text-align:center;">'
        f'    <div style="font-size:0.72rem;color:{_MUTED};'
        f'      text-transform:uppercase;">95% Band</div>'
        f'    <div style="font-size:1rem;font-weight:600;'
        f'      color:{_NAVY};">'
        f"      {lo_display}\u2009\u2013\u2009{hi_display}</div>"
        f"  </div>"
        # Drift rate
        f'  <div style="background:{_CREAM};padding:0.6rem;'
        f'    border-radius:4px;text-align:center;">'
        f'    <div style="font-size:0.72rem;color:{_MUTED};'
        f'      text-transform:uppercase;">Drift / Decade</div>'
        f'    <div style="font-size:1rem;font-weight:600;'
        f'      color:{_NAVY};">{drift:.3f}</div>'
        f"  </div>"
        f"</div>"
    )

    return _section(
        "Temporal Analysis",
        timeline_html + stats_html,
    )
