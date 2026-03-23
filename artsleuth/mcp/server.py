"""
MCP (Model Context Protocol) server for ArtSleuth.

Exposes ArtSleuth's analytical capabilities as MCP tools, allowing
AI assistants (Claude, GPT, and others) to perform art-historical
analysis within a conversational context.

Launch with ``artsleuth server`` or wire it into Claude Desktop via
the MCP config.  Four tools: full analysis, style classification,
side-by-side comparison, and forgery screening.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_server() -> Any:
    """Create and configure the ArtSleuth MCP server.

    Returns
    -------
    mcp.Server
        A configured MCP server instance ready to run.

    Raises
    ------
    ImportError
        If the ``mcp`` package is not installed (install with
        ``pip install artsleuth[mcp]``).
    """
    try:
        from mcp.server import Server
        from mcp.types import Tool, TextContent
    except ImportError:
        raise ImportError(
            "The MCP server requires the 'mcp' package. "
            "Install it with: pip install artsleuth[mcp]"
        )

    server = Server("artsleuth")

    # --- Tool Definitions ---------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="analyze_artwork",
                description=(
                    "Run the full ArtSleuth analysis pipeline on an artwork image. "
                    "Returns style classification, brushstroke analysis, artist "
                    "attribution, and optional forgery screening."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to the artwork image file.",
                        },
                        "reference_artist": {
                            "type": "string",
                            "description": (
                                "Optional artist name for forgery screening. "
                                "If provided, the painting is checked for anomalies "
                                "relative to this artist's reference corpus."
                            ),
                        },
                    },
                    "required": ["image_path"],
                },
            ),
            Tool(
                name="classify_style",
                description=(
                    "Classify an artwork by historical period, art-historical "
                    "school, and subject genre."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to the artwork image file.",
                        },
                    },
                    "required": ["image_path"],
                },
            ),
            Tool(
                name="compare_works",
                description=(
                    "Compare two artworks for stylistic similarity based on "
                    "embedding-space proximity. Useful as a screening tool, "
                    "but does not confirm shared authorship on its own."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_path_a": {
                            "type": "string",
                            "description": "Path to the first artwork.",
                        },
                        "image_path_b": {
                            "type": "string",
                            "description": "Path to the second artwork.",
                        },
                    },
                    "required": ["image_path_a", "image_path_b"],
                },
            ),
            Tool(
                name="detect_anomalies",
                description=(
                    "Screen a painting for statistical anomalies relative to "
                    "a reference artist's corpus, flagging potential forgeries."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to the artwork image file.",
                        },
                        "reference_artist": {
                            "type": "string",
                            "description": "Artist whose corpus defines normality.",
                        },
                    },
                    "required": ["image_path", "reference_artist"],
                },
            ),
        ]

    # --- Tool Handlers ------------------------------------------------------

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            result = _dispatch(name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as exc:
            logger.exception("Tool '%s' failed.", name)
            return [TextContent(type="text", text=f"Error: {exc}")]

    return server


def _dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Route a tool call to the appropriate ArtSleuth function."""
    from artsleuth.config import AnalysisConfig

    config = AnalysisConfig()

    if name == "analyze_artwork":
        return _handle_analyze(arguments, config)
    elif name == "classify_style":
        return _handle_classify(arguments, config)
    elif name == "compare_works":
        return _handle_compare(arguments, config)
    elif name == "detect_anomalies":
        return _handle_detect(arguments, config)
    else:
        raise ValueError(f"Unknown tool: {name}")


def _handle_analyze(args: dict[str, Any], config: Any) -> dict[str, Any]:
    """Handle ``analyze_artwork`` tool calls."""
    from artsleuth.core.pipeline import run_pipeline

    result = run_pipeline(
        args["image_path"],
        config=config,
        reference_artist=args.get("reference_artist"),
    )
    return {
        "summary": result.summary(),
        "style": {
            "period": {"label": result.style.period.label, "confidence": result.style.period.confidence},
            "school": {"label": result.style.school.label, "confidence": result.style.school.confidence},
            "genre": {"label": result.style.technique.label, "confidence": result.style.technique.confidence},
        },
        "attribution": {
            "consensus_artist": result.attribution.consensus_artist,
            "confidence": result.attribution.consensus_confidence,
            "multi_hand": result.attribution.multi_hand_flag,
        },
    }


def _handle_classify(args: dict[str, Any], config: Any) -> dict[str, Any]:
    """Handle ``classify_style`` tool calls."""
    from PIL import Image

    from artsleuth.core.style import StyleClassifier

    image = Image.open(args["image_path"]).convert("RGB")
    classifier = StyleClassifier(config)
    report = classifier.classify(image)

    return {
        "period": {"label": report.period.label, "confidence": report.period.confidence, "top_k": report.period.top_k},
        "school": {"label": report.school.label, "confidence": report.school.confidence, "top_k": report.school.top_k},
        "genre": {"label": report.technique.label, "confidence": report.technique.confidence, "top_k": report.technique.top_k},
    }


def _handle_compare(args: dict[str, Any], config: Any) -> dict[str, Any]:
    """Handle ``compare_works`` tool calls."""
    from PIL import Image

    from artsleuth.core.style import StyleClassifier

    classifier = StyleClassifier(config)
    img_a = Image.open(args["image_path_a"]).convert("RGB")
    img_b = Image.open(args["image_path_b"]).convert("RGB")

    report_a = classifier.classify(img_a)
    report_b = classifier.classify(img_b)

    import numpy as np

    similarity = float(
        np.dot(report_a.embedding, report_b.embedding)
        / (np.linalg.norm(report_a.embedding) * np.linalg.norm(report_b.embedding) + 1e-12)
    )

    return {
        "similarity": similarity,
        "interpretation": (
            "High embedding similarity — shared stylistic features, but does not confirm shared authorship"
            if similarity > 0.85
            else "Moderate similarity — some shared stylistic traits"
            if similarity > 0.65
            else "Low similarity — some overlapping features"
            if similarity > 0.45
            else "Minimal similarity — distinct stylistic profiles"
        ),
        "work_a": {"period": report_a.period.label, "school": report_a.school.label, "genre": report_a.technique.label},
        "work_b": {"period": report_b.period.label, "school": report_b.school.label, "genre": report_b.technique.label},
    }


def _handle_detect(args: dict[str, Any], config: Any) -> dict[str, Any]:
    """Handle ``detect_anomalies`` tool calls."""
    from PIL import Image

    from artsleuth.core.forgery import ForgeryDetector

    image = Image.open(args["image_path"]).convert("RGB")
    detector = ForgeryDetector(config)
    report = detector.detect(image, reference_artist=args["reference_artist"])

    return {
        "anomaly_score": report.anomaly_score,
        "is_flagged": report.is_flagged,
        "reference_artist": report.reference_artist,
        "indicators": [
            {"feature": ind.feature_name, "z_score": ind.z_score, "description": ind.description}
            for ind in report.indicators
        ],
    }
