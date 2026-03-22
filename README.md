<div align="center">

<br>

<img src="assets/banner.png" alt="ArtSleuth Banner" width="100%" />

<br><br>

[![Python](https://img.shields.io/badge/Python-3.10+-9DC0D8?style=for-the-badge&labelColor=1A2E48&logo=python&logoColor=9DC0D8)](https://www.python.org/)&nbsp;
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-9DC0D8?style=for-the-badge&labelColor=1A2E48&logo=pytorch&logoColor=9DC0D8)](https://pytorch.org/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-9DC0D8?style=for-the-badge&labelColor=1A2E48&logo=huggingface&logoColor=9DC0D8)](https://huggingface.co/)&nbsp;
[![Demo](https://img.shields.io/badge/Demo-Live-d4af37?style=for-the-badge&labelColor=1A2E48)](https://ladyfaye1998.github.io/ArtSleuth/)&nbsp;
[![MCP](https://img.shields.io/badge/MCP-Server-9DC0D8?style=for-the-badge&labelColor=1A2E48)](https://modelcontextprotocol.io/)&nbsp;
[![License](https://img.shields.io/badge/License-MIT-9DC0D8?style=for-the-badge&labelColor=1A2E48)](LICENSE)

<br>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Lora&weight=500&size=22&pause=2800&color=9DC0D8&center=true&vCenter=true&width=680&lines=Brushstroke+forensics+%C2%B7+Style+attribution+%C2%B7+Forgery+detection;Cross-attention+backbone+fusion+%C2%B7+Temporal+drift+modelling;Where+connoisseurship+meets+computation)](https://github.com/ladyFaye1998/ArtSleuth)

<br>

</div>

---

<br>

### ✦ About

**ArtSleuth** is a computational art-analysis framework that formalises what connoisseurs have done for centuries — examining the physical evidence a painter leaves on a canvas — using machine learning.

Brushstroke directionality, impasto relief, palette temperature, the habitual gestures that reside in the least-scrutinised passages of a painting — drapery folds, background foliage, the rendering of earlobes. These are the signals that distinguish one hand from another, and they map naturally onto what self-supervised vision transformers learn to encode.

<br>

<div align="center">

| | Capability | Method |
|:---:|:---|:---|
| ![Brushstroke](https://img.shields.io/badge/-Brushstroke_Analysis-1A2E48?style=flat-square) | Stroke orientation, coherence, energy, curvature with patch-level clustering | Structure tensor decomposition via DINOv2 |
| ![Style](https://img.shields.io/badge/-Style_Classification-1A2E48?style=flat-square) | Period, school, and technique prediction | CLIP embeddings through learned linear heads |
| ![Attribution](https://img.shields.io/badge/-Artist_Attribution-1A2E48?style=flat-square) | Embedding-space comparison with temporal plausibility scoring | Cosine similarity with GP-based date estimation |
| ![Workshop](https://img.shields.io/badge/-Workshop_Decomposition-1A2E48?style=flat-square) | Bayesian inference of distinct hands in collaborative paintings | Dirichlet process Gaussian mixture model |
| ![Forgery](https://img.shields.io/badge/-Forgery_Detection-1A2E48?style=flat-square) | One-class anomaly scoring with adversarial robustness testing | Mahalanobis distance plus historical forgery simulation |
| ![Fusion](https://img.shields.io/badge/-Cross--Attention_Fusion-1A2E48?style=flat-square) | Style-guided patch attention across dual backbones | Multi-head cross-attention (CLIP Q, DINOv2 KV) |
| ![Temporal](https://img.shields.io/badge/-Temporal_Drift-1A2E48?style=flat-square) | Models how an artist's style evolves over decades | Gaussian process regression in embedding space |
| ![Explainability](https://img.shields.io/badge/-Explainability-1A2E48?style=flat-square) | Visual heatmaps showing where the model looks and why | Grad-CAM and attention rollout |

</div>

<br>

---

<br>

### ✦ What's Novel

ArtSleuth introduces four contributions not found in existing art-analysis frameworks:

1. **Style-Guided Cross-Attention Fusion** — CLIP's semantic understanding directs DINOv2's patch-level attention via multi-head cross-attention with learned temperature, producing fused features neither backbone achieves alone.

2. **Temporal Style Drift Modelling** — Gaussian process regression over time-stamped reference embeddings captures how an artist's hand evolves across decades, adjusting attribution scores for temporal plausibility.

3. **Hierarchical Workshop Decomposition** — A Dirichlet process Gaussian mixture model automatically infers the number of distinct hands in a painting, replacing flat k-means with art-historically grounded probabilistic clustering.

4. **Adversarial Forgery Robustness** — Stress-tests detection against simulated historical forgery techniques (artificial aging, style transfer perturbation, material anachronism) at multiple severity levels.

<br>

---

<br>

### ✦ Quick Start

```bash
pip install artsleuth
```

<br>

**Python**

```python
import artsleuth

result = artsleuth.analyze("judith_slaying_holofernes.jpg")
print(result.summary())

explanation = result.explain()
explanation.save("analysis_overlay.png")
```

<br>

**CLI**

```bash
artsleuth analyze painting.jpg
artsleuth style painting.jpg --top-k 5
artsleuth compare painting_a.jpg painting_b.jpg
artsleuth workshop painting.jpg
artsleuth robustness painting.jpg -r "Artemisia Gentileschi"
artsleuth benchmark --backbone dinov2 --backbone fusion
artsleuth demo
```

<br>

---

<br>

### ✦ Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#1A2E48', 'primaryTextColor': '#F0F0F0', 'primaryBorderColor': '#9DC0D8', 'lineColor': '#9DC0D8', 'secondaryColor': '#1A2E48', 'tertiaryColor': '#1A2E48', 'edgeLabelBackground': '#0D1117', 'clusterBkg': '#0D1117', 'clusterBorder': '#9DC0D8', 'titleColor': '#9DC0D8'}}}%%

graph TD
    Input["Artwork Image"] --> Preprocess

    subgraph prep ["Preprocessing"]
        Preprocess["Varnish Correction"] --> Crack["Craquelure Suppression"]
        Crack --> Canvas["Canvas Texture Filtering"]
    end

    prep --> Patches["Patch Extraction"]
    prep --> FullImage["Full-Image Encoding"]

    Patches --> DINO["DINOv2 ViT-S/14"]
    FullImage --> CLIPEnc["CLIP ViT-B/32"]

    DINO --> CrossAttn["Style-Guided Cross-Attention"]
    CLIPEnc --> CrossAttn

    CrossAttn --> Brushstroke["Brushstroke Analysis"]
    CrossAttn --> WorkshopNode["Workshop Decomposition"]
    CLIPEnc --> Style["Style Classification"]
    CrossAttn --> Attribution["Attribution Scoring"]
    Attribution --> Temporal["Temporal Drift Model"]

    Brushstroke --> Fusion["Feature Fusion"]
    Style --> Fusion
    Fusion --> Forgery["Forgery Detection"]
    Fusion --> Adversarial["Adversarial Robustness"]
    Fusion --> Explain["Explainability Engine"]

    Forgery --> Report["Analysis Report"]
    Attribution --> Report
    Temporal --> Report
    Style --> Report
    Brushstroke --> Report
    WorkshopNode --> Report
    Explain --> Report

    Report --> WebUI["Web UI"]
    Report --> CLI["CLI"]
    Report --> MCP["MCP Server"]
```

<br>

<div align="center">

| Backbone | Strength | Used For |
|:---|:---|:---|
| **DINOv2** · ViT-S/14 | Fine-grained texture and structure | Brushstroke analysis · cross-attention K/V |
| **CLIP** · ViT-B/32 | Semantic-stylistic understanding | Style classification · cross-attention Q |
| **Fusion** · Cross-Attention | Style-aware structural features | Attribution · forgery · workshop decomposition |

</div>

<br>

---

<br>

### ✦ Benchmark

Linear probe evaluation on the full [WikiArt](https://huggingface.co/datasets/huggan/wikiart) dataset (81 444 images, logistic regression, 80/20 split, seed 42):

<div align="center">

| Backbone | Style Acc | Style F1 | Artist Acc | Artist Top-5 | Genre Acc |
|:---|:---:|:---:|:---:|:---:|:---:|
| DINOv2 · ViT-S/14 | 55.4 % | 0.522 | 65.0 % | 90.8 % | 68.7 % |
| CLIP · ViT-B/32 | 62.4 % | 0.603 | 70.3 % | 93.8 % | 71.7 % |
| Fusion · frozen | 62.2 % | 0.589 | 70.3 % | 94.0 % | 71.7 % |
| **Fusion · fine-tuned** | **63.6 %** | **0.616** | **72.7 %** | **94.7 %** | **72.3 %** |

</div>

<sub>All metrics macro-averaged across 27 styles, 129 artists, and 11 genres. The fine-tuned fusion head (15 epochs, AdamW, cross-entropy) consistently outperforms both individual backbones. Reproducible notebook on [Kaggle](https://www.kaggle.com/ladyfaye/artsleuth-full-pipeline-benchmark-fine-tune).</sub>

<br>

Reproduce locally:

```bash
pip install artsleuth[benchmarks]
artsleuth benchmark --device cuda
```

<br>

---

<br>

### ✦ Web Demo

An interactive Gradio interface with five analysis tabs: full pipeline, side-by-side comparison, workshop decomposition, temporal dating, and benchmark dashboard.

```bash
pip install artsleuth[web]
artsleuth demo
```

Or try the hosted version at [ladyfaye1998.github.io/ArtSleuth](https://ladyfaye1998.github.io/ArtSleuth/).

<br>

---

<br>

### ✦ MCP Server

ArtSleuth ships as an [MCP](https://modelcontextprotocol.io/) server, enabling AI assistants to perform art analysis conversationally.

```bash
artsleuth server
```

<br>

<div align="center">

| Tool | Description |
|:---|:---|
| `analyze_artwork` | Full analysis pipeline |
| `classify_style` | Period, school, technique classification |
| `compare_works` | Side-by-side stylistic comparison |
| `detect_anomalies` | Forgery screening against a reference corpus |

</div>

<br>

<details>
<summary>&nbsp;Claude Desktop configuration</summary>

<br>

```json
{
  "mcpServers": {
    "artsleuth": {
      "command": "artsleuth",
      "args": ["server"]
    }
  }
}
```

</details>

<br>

---

<br>

### ✦ Repository Structure

```
ArtSleuth/
├── artsleuth/
│   ├── core/                  # Analysis engines
│   │   ├── brushstroke.py     #   Brushstroke pattern extraction
│   │   ├── style.py           #   Style classification
│   │   ├── attribution.py     #   Artist attribution scoring
│   │   ├── forgery.py         #   Anomaly-based forgery detection
│   │   ├── explainability.py  #   GradCAM & attention overlays
│   │   ├── temporal.py        #   Temporal style drift (GP)
│   │   ├── workshop.py        #   Bayesian workshop decomposition
│   │   ├── adversarial.py     #   Adversarial robustness testing
│   │   └── pipeline.py        #   Unified analysis orchestrator
│   ├── models/                # Backbone & head architectures
│   │   ├── backbones.py       #   DINOv2 & CLIP loaders
│   │   ├── fusion.py          #   Cross-attention backbone fusion
│   │   ├── heads.py           #   Task-specific linear heads
│   │   └── registry.py        #   HuggingFace model registry
│   ├── preprocessing/         # Art-specific transforms
│   │   ├── transforms.py      #   Varnish, crack, canvas correction
│   │   └── patches.py         #   Grid, salient, adaptive extraction
│   ├── benchmarks/            # Evaluation suite
│   │   ├── wikiart.py         #   WikiArt dataset + linear probes
│   │   └── evaluate.py        #   Multi-backbone comparison runner
│   ├── mcp/                   # MCP server
│   │   └── server.py          #   Tool definitions & handlers
│   ├── cli/                   # Command-line interface
│   │   └── main.py            #   Click-based CLI
│   └── utils/                 # Shared utilities
│       ├── visualization.py   #   Publication-quality figures
│       └── io.py              #   Image loading & saving
├── web/                       # Gradio web demo
│   ├── app.py                 #   Main application (5 tabs)
│   ├── theme.py               #   Custom ArtSleuth theme
│   └── components.py          #   Reusable UI builders
├── tests/                     # Pytest suite (9 test modules)
├── examples/                  # Jupyter notebooks
├── docs/                      # Methodology & guides
├── assets/                    # Visual assets
└── index.html                 # GitHub Pages landing site
```

<br>

---

<br>

### ✦ Development

```bash
git clone https://github.com/ladyFaye1998/ArtSleuth.git
cd ArtSleuth
pip install -e ".[all]"

pytest
ruff check .
mypy artsleuth
```

<br>

---

<br>

### ✦ Methodology

ArtSleuth draws on two traditions:

**Art history** — Giovanni Morelli's observation (1890) that an artist's most characteristic habits reside in the least-conscious passages. Bernard Berenson's refinement of this into systematic connoisseurship. The workshop-attribution methodology developed for the Gentileschi debate, where master and assistants each contribute recognisable passages to a shared canvas.

**Computer science** — Self-supervised vision transformers (Caron et al., 2021; Oquab et al., 2024) that learn rich visual features without task-specific labels. Contrastive vision-language models (Radford et al., 2021) that ground visual concepts in linguistic semantics. Cross-attention fusion (Vaswani et al., 2017; Jose et al., 2025) for multi-modal feature alignment. Dirichlet process mixtures (Blei & Jordan, 2006) for non-parametric clustering. Gaussian processes (Rasmussen & Williams, 2006) for temporal modelling.

The two complement each other: art history provides the *questions*; machine learning provides a *scale* of analysis that would be impractical by eye alone.

See [`docs/methodology.md`](docs/methodology.md) for the full technical discussion.

<br>

---

<br>

### ✦ Citation

```bibtex
@software{lesin2026artsleuth,
  author    = {Lesin, Danielle},
  title     = {{ArtSleuth}: {AI} Art Forensics \& Analysis Framework},
  year      = {2026},
  url       = {https://github.com/ladyFaye1998/ArtSleuth},
  license   = {MIT}
}
```

<br>

---

<br>

### ✦ Contributing

Contributions are welcome from art historians, ML researchers, conservators, and anyone interested in computational approaches to cultural heritage.

<br>

<div align="center">

| Area | What's Needed |
|:---|:---|
| **Reference corpora** | Curated, well-attributed image sets for specific artists or periods |
| **Temporal references** | Dated works for training the temporal style drift model |
| **Model improvements** | Better backbones, training strategies, evaluation benchmarks |
| **Art-historical review** | Ensuring taxonomy, terminology, and methodology stay sound |
| **Web UI** | Gradio component improvements, accessibility, visualisation refinements |
| **Bug reports** | [Open an issue](https://github.com/ladyFaye1998/ArtSleuth/issues) with reproduction steps |

</div>

<br>

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

<br>

---

<br>

<div align="center">

<sub>Built with 🫖 by <a href="https://github.com/ladyFaye1998">Danielle Lesin</a> · Where connoisseurship meets computation</sub>

<br><br>

</div>
