"""Reusable UI component builders for the ArtSleuth Gradio app.

Every ``format_*`` function returns a self-contained HTML string that
uses CSS classes defined in ``web/theme.py`` CUSTOM_CSS so it renders
correctly inside Gradio's dark-mode ``.prose`` wrapper.
"""
from __future__ import annotations

import html as _html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from artsleuth.core.attribution import AttributionReport
    from artsleuth.core.forgery import ForgeryReport
    from artsleuth.core.style import StyleReport
    from artsleuth.core.workshop import WorkshopReport

# Bar accent colours per axis
_CLR_GOLD = "#d4a843"
_CLR_BLUE = "#68b5d5"
_CLR_ROSE = "#f0768a"
_CLR_GREEN = "#4ade80"
_CLR_PURPLE = "#a78bfa"
_CLR_AMBER = "#fbbf24"

_HAND_COLORS = [_CLR_GOLD, _CLR_ROSE, _CLR_BLUE, _CLR_GREEN, _CLR_PURPLE, _CLR_AMBER]


def _esc(text: str) -> str:
    return _html.escape(str(text))


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _bar(label: str, value: float, color: str = _CLR_GOLD) -> str:
    pct = min(max(value * 100, 0), 100)
    return (
        '<div class="as-bar-wrap">'
        '<div class="as-bar-top">'
        f"<span>{_esc(label)}</span>"
        f'<span class="as-pct">{pct:.1f}%</span>'
        "</div>"
        '<div class="as-bar-track">'
        f'<div class="as-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>'
        "</div></div>"
    )


def _card(title: str, body: str) -> str:
    return (
        '<div class="as-card">'
        f"<h3>{_esc(title)}</h3>"
        f"{body}"
        "</div>"
    )


def _badge(text: str, bg: str = _CLR_ROSE, fg: str = "#0c0a09") -> str:
    return (
        f'<span class="as-badge" style="background:{bg};color:{fg};">'
        f"{_esc(text)}</span>"
    )


# ── Public API ──────────────────────────────────────────────────────


def build_header() -> str:
    from web.theme import HEADER_HTML
    return HEADER_HTML


def build_footer() -> str:
    from web.theme import FOOTER_HTML
    return FOOTER_HTML


def format_style_report(style_report: StyleReport) -> str:
    axes = [
        ("Period", style_report.period, _CLR_GOLD),
        ("School", style_report.school, _CLR_BLUE),
        ("Genre", style_report.technique, _CLR_ROSE),
    ]
    parts: list[str] = []
    for axis_name, pred, color in axes:
        bars = "".join(_bar(label, conf, color) for label, conf in pred.top_k)
        parts.append(_card(
            f"{axis_name}: {_esc(pred.label)} ({_pct(pred.confidence)})",
            bars,
        ))
    return "".join(parts)


def format_attribution_report(attribution_report: AttributionReport) -> str:
    rows: list[str] = []
    for rank, cand in enumerate(attribution_report.candidates, start=1):
        lo, hi = cand.confidence_interval
        bar = _bar(
            f"#{rank} {cand.artist}", cand.score,
            _CLR_GOLD if rank == 1 else _CLR_BLUE,
        )
        features = ", ".join(cand.supporting_features) or "\u2014"
        rows.append(
            f"{bar}"
            f'<div style="font-size:0.78rem;color:#78716c;margin:-2px 0 10px 4px;">'
            f"  95% CI: [{lo:.2f}, {hi:.2f}] &middot; {_esc(features)}"
            "</div>"
        )

    header_extra = ""
    if attribution_report.multi_hand_flag:
        header_extra = f" {_badge('MULTI-HAND', bg=_CLR_AMBER, fg='#0c0a09')}"

    consensus = (
        '<div style="font-size:0.92rem;margin-bottom:0.8rem;">'
        f"<strong>Consensus:</strong> "
        f"{_esc(attribution_report.consensus_artist)} "
        f"({_pct(attribution_report.consensus_confidence)}) "
        f"{header_extra}"
        "</div>"
    )
    return _card("Attribution Candidates", consensus + "".join(rows))


def format_forgery_gauge(forgery_report: ForgeryReport) -> str:
    score = forgery_report.anomaly_score
    pct = score * 100

    if score < 0.4:
        ring_color, verdict = _CLR_GREEN, "LOW RISK"
    elif score < 0.7:
        ring_color, verdict = _CLR_AMBER, "MODERATE"
    else:
        ring_color, verdict = "#f87171", "HIGH RISK"

    flag = _badge("FLAGGED", bg="#f87171") if forgery_report.is_flagged else ""

    gauge = (
        '<div style="text-align:center;">'
        f'<div style="width:140px;height:140px;border-radius:50%;margin:0 auto;'
        f"background:conic-gradient({ring_color} 0deg {pct*3.6:.1f}deg, "
        f'rgba(255,255,255,0.06) {pct*3.6:.1f}deg 360deg);'
        f'box-shadow:0 0 30px rgba(0,0,0,0.3);">'
        '<div style="width:104px;height:104px;border-radius:50%;'
        f"background:#1c1917;position:relative;top:18px;left:18px;"
        f'display:flex;align-items:center;justify-content:center;flex-direction:column;">'
        f'<span style="font-family:\'Playfair Display\',serif;font-size:1.8rem;'
        f'font-weight:700;color:#faf9f6;">{score:.2f}</span>'
        f'<span style="font-size:0.62rem;color:{ring_color};text-transform:uppercase;'
        f'letter-spacing:0.08em;font-weight:600;">{verdict}</span>'
        "</div></div>"
        f'<div style="margin-top:0.6rem;">{flag}</div>'
        f'<div style="font-size:0.78rem;color:#78716c;margin-top:0.3rem;">'
        f"Reference: {_esc(forgery_report.reference_artist)}</div>"
        "</div>"
    )

    indicator_rows = ""
    for ind in forgery_report.indicators:
        z_clr = "#f87171" if ind.z_score > 3 else _CLR_AMBER if ind.z_score > 2 else _CLR_GREEN
        indicator_rows += (
            f'<tr><td style="padding:5px 8px;">{_esc(ind.feature_name)}</td>'
            f'<td style="padding:5px 8px;color:{z_clr};font-weight:600;">'
            f"z\u202f=\u202f{ind.z_score:.1f}</td>"
            f'<td style="padding:5px 8px;color:#78716c;font-size:0.8rem;">'
            f"{_esc(ind.description)}</td></tr>"
        )

    table = ""
    if indicator_rows:
        table = (
            '<table style="width:100%;border-collapse:collapse;margin-top:1rem;'
            'font-size:0.84rem;">'
            '<tr style="border-bottom:1px solid rgba(212,168,67,0.15);">'
            '<th style="text-align:left;padding:5px 8px;">Feature</th>'
            '<th style="text-align:left;padding:5px 8px;">Z-Score</th>'
            '<th style="text-align:left;padding:5px 8px;">Detail</th></tr>'
            f"{indicator_rows}</table>"
        )

    return _card("Forgery Screening", gauge + table)


def format_workshop_report(workshop_report: WorkshopReport) -> str:
    ws_badge = ""
    if workshop_report.is_workshop:
        ws_badge = _badge("WORKSHOP PRODUCTION", bg=_CLR_AMBER, fg="#0c0a09")

    summary = (
        '<div style="font-size:0.92rem;margin-bottom:0.8rem;">'
        f"<strong>Detected hands:</strong> {workshop_report.num_hands} {ws_badge}"
        "</div>"
    )

    cards: list[str] = []
    for idx, hand in enumerate(workshop_report.assignments):
        color = _HAND_COLORS[idx % len(_HAND_COLORS)]
        label = hand.label.replace("_", " ").title()
        extent = _bar("Spatial extent", hand.spatial_extent, color)
        conf = _bar("Confidence", hand.confidence, color)
        stats = (
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;'
            'font-size:0.82rem;color:#78716c;margin-top:6px;">'
            f"<span>Patches: {hand.patch_count}</span>"
            f"<span>Coherence: {hand.mean_coherence:.3f}</span>"
            f"<span>Energy: {hand.mean_energy:.3f}</span>"
            f"<span>Hand ID: {hand.hand_id}</span>"
            "</div>"
        )
        cards.append(
            f'<div style="border-left:3px solid {color};'
            f"padding:0.5rem 0.75rem;margin:0.5rem 0;"
            f'background:rgba(255,255,255,0.02);border-radius:0 8px 8px 0;">'
            f'<div style="font-weight:600;color:#e7e5e4;font-size:0.95rem;">'
            f"{_esc(label)}</div>"
            f"{extent}{conf}{stats}"
            "</div>"
        )

    return _card("Workshop Decomposition", summary + "".join(cards))


def format_temporal_prediction(temporal_prediction) -> str:
    year = temporal_prediction.estimated_year
    lo, hi = temporal_prediction.confidence_band
    score = temporal_prediction.temporal_score
    drift = temporal_prediction.drift_rate

    score_cls = (
        "as-score-good" if score > 0.7
        else "as-score-mid" if score > 0.4
        else "as-score-bad"
    )

    return _card("Temporal Analysis", (
        '<div class="as-date">'
        f'<div class="as-date__year">c.\u2009{year:.0f}</div>'
        '<div class="as-date__label">Estimated Date</div>'
        '<div class="as-date__stats">'
        '<div style="text-align:center">'
        '<div class="as-date__stat-label">95% Band</div>'
        f'<div class="as-date__stat-value">{lo:.0f}\u2013{hi:.0f}</div>'
        "</div>"
        '<div style="text-align:center">'
        '<div class="as-date__stat-label">Plausibility</div>'
        f'<div class="as-date__stat-value {score_cls}">{score:.0%}</div>'
        "</div>"
        '<div style="text-align:center">'
        '<div class="as-date__stat-label">Drift / Decade</div>'
        f'<div class="as-date__stat-value">{drift:.3f}</div>'
        "</div></div></div>"
    ))
