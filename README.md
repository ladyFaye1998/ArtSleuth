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

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Lora&weight=500&size=22&pause=2800&color=9DC0D8&center=true&vCenter=true&width=680&lines=Brushstroke+analysis+%C2%B7+Style+attribution+%C2%B7+Anomaly+screening;Dual-backbone+feature+fusion+%C2%B7+Temporal+drift+modelling;Where+connoisseurship+meets+computation)](https://github.com/ladyFaye1998/ArtSleuth)

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
| ![Brushstroke](https://img.shields.io/badge/-Brushstroke_Analysis-1A2E48?style=flat-square) | Stroke orientation, coherence, energy, curvature with patch-level clustering | Structure tensor on image gradients + DINOv2 patch embeddings |
| ![Style](https://img.shields.io/badge/-Style_Classification-1A2E48?style=flat-square) | Period, school, and technique prediction | CLIP embeddings through learned linear heads |
| ![Attribution](https://img.shields.io/badge/-Artist_Attribution-1A2E48?style=flat-square) | Embedding-space comparison with temporal plausibility scoring | Cosine similarity with GP-based date estimation |
| ![Workshop](https://img.shields.io/badge/-Workshop_Decomposition-1A2E48?style=flat-square) | Bayesian inference of distinct hands in collaborative paintings | Dirichlet process Gaussian mixture model |
| ![Forgery](https://img.shields.io/badge/-Anomaly_Screening-1A2E48?style=flat-square) | One-class anomaly scoring with adversarial robustness testing | Mahalanobis distance plus historical forgery simulation |
| ![Fusion](https://img.shields.io/badge/-Dual--Backbone_Fusion-1A2E48?style=flat-square) | Complementary features from two vision transformers | Concatenation at inference; cross-attention available for training |
| ![Temporal](https://img.shields.io/badge/-Temporal_Drift-1A2E48?style=flat-square) | Models how an artist's style evolves over decades | Gaussian process regression in embedding space |
| ![Explainability](https://img.shields.io/badge/-Explainability-1A2E48?style=flat-square) | Visual heatmaps highlighting regions the model considers salient | Gradient-based saliency maps |

</div>

<br>

---

<br>

### ✦ What's Novel

ArtSleuth combines several techniques that are typically studied in isolation:

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
    Input["Artwork Image"] --> Resize["Resize · Crop · Normalise"]

    Resize --> Patches["Patch Extraction"]
    Resize --> FullImage["Full-Image Encoding"]

    Patches --> DINO["DINOv2"]
    FullImage --> CLIPEnc["CLIP"]

    DINO --> Brushstroke["Brushstroke Analysis"]
    DINO --> WorkshopNode["Workshop Decomposition"]
    CLIPEnc --> Style["Style Classification"]

    DINO --> Concat["Feature Concatenation"]
    CLIPEnc --> Concat

    Concat --> Attribution["Attribution Scoring"]
    Attribution --> Temporal["Temporal Drift Model"]

    Concat --> Forgery["Forgery Detection"]
    Concat --> Adversarial["Adversarial Robustness"]

    DINO --> Explain["Saliency Maps"]

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

<sub>The inference pipeline concatenates CLIP + DINOv2 embeddings. Cross-attention fusion is available as a training-time architecture (used in the benchmark fine-tuning) but is **not** part of the default inference path.</sub>

<br>

<div align="center">

| Backbone | Strength | Used For |
|:---|:---|:---|
| **DINOv2** | Fine-grained texture and structure | Brushstroke analysis · patch embeddings |
| **CLIP** | Semantic-stylistic understanding | Style classification · style embeddings |
| **Concat** | Complementary feature combination | Attribution · forgery detection |

</div>

<sub>Default backbone sizes: DINOv2 ViT-B/14 + CLIP ViT-L/14. Pretrained classification heads (style and genre) are included. Pass `--backbone-size small` for faster inference on constrained hardware, or `--backbone-size large` for the highest-quality embeddings.</sub>

<br>

---

<br>

### ✦ Benchmark

Linear probe and end-to-end evaluation on the full [WikiArt](https://huggingface.co/datasets/huggan/wikiart) dataset (81 444 images, 80/20 split, seed 42):

<div align="center">

| Backbone | Style Acc | Style F1 | Artist Acc | Artist Top-5 | Genre Acc |
|:---|:---:|:---:|:---:|:---:|:---:|
| DINOv2 · ViT-B/14 | 57.5 % | 0.553 | 64.7 % | 90.9 % | 71.0 % |
| CLIP · ViT-L/14 | 67.1 % | 0.656 | 74.6 % | 95.9 % | 75.0 % |
| Fusion · frozen | 65.0 % | 0.633 | 71.0 % | 94.2 % | 74.2 % |
| Fusion · fine-tuned | 71.6 % | 0.703 | 77.8 % | 96.2 % | 75.1 % |
| **Fusion · e2e** | **72.7 %** | — | **79.0 %** | **96.9 %** | **76.6 %** |

</div>

<sub>Top four rows: logistic-regression linear probes (macro-averaged across 27 styles, 129 artists, 11 genres). Bottom row: end-to-end classification heads trained jointly with the fusion backbone. Fine-tuning partially unfreezes the last 3 transformer blocks of each backbone, uses multi-task CE + supervised contrastive loss, AdamW with cosine annealing, and mixed-precision training (5 epochs, effective batch 64). Reproducible notebook on [Kaggle](https://www.kaggle.com/ladyfaye/artsleuth-benchmark).</sub>

<br>

Reproduce locally:

```bash
pip install artsleuth[benchmarks]
artsleuth benchmark --device cuda --backbone-size base
```

<br>

<details>
<summary>&nbsp;Forgery detection validation (one-class authentication)</summary>

<br>

For each of 125 named artists (≥ 80 works, excluding the catch-all "Unknown Artist" category), we fit a Mahalanobis-distance reference model from 80 % of their authenticated WikiArt works, then test whether held-out genuine paintings score lower (closer to the reference distribution) than impostor paintings by other artists. ROC-AUC = 1.0 means perfect separation; 0.5 means chance.

<div align="center">

| | DINOv2 ViT-B/14 | CLIP ViT-L/14 | Fused (concat) |
|:---|:---:|:---:|:---:|
| **Mean AUC** | 0.867 | 0.958 | 0.891 |
| **Median AUC** | 0.869 | 0.966 | 0.902 |
| AUC ≥ 0.95 | 25 / 125 artists | 98 / 125 artists | 40 / 125 artists |
| AUC ≥ 0.90 | 56 / 125 artists | 119 / 125 artists | 66 / 125 artists |

</div>

<br>

Top 15 and bottom 5 by fused AUC:

<div align="center">

| Artist | Works | DINOv2 | CLIP | Fused |
|:---|:---:|:---:|:---:|:---:|
| Sam Francis | 317 | 1.000 | 1.000 | 1.000 |
| Gene Davis | 155 | 0.998 | 1.000 | 1.000 |
| Michelangelo | 129 | 0.996 | 1.000 | 1.000 |
| Antoine Blanchard | 170 | 1.000 | 1.000 | 1.000 |
| El Greco | 159 | 0.994 | 0.998 | 0.999 |
| Édouard Cortès | 214 | 0.998 | 1.000 | 0.999 |
| Utagawa Kuniyoshi | 199 | 0.981 | 1.000 | 0.994 |
| Fra Angelico | 167 | 0.988 | 0.999 | 0.992 |
| Anthony van Dyck | 163 | 0.979 | 1.000 | 0.989 |
| Juan Gris | 196 | 0.983 | 0.999 | 0.988 |
| Ivan Aivazovsky | 577 | 0.980 | 0.998 | 0.986 |
| Georges Braque | 208 | 0.975 | 0.989 | 0.985 |
| Gustave Loiseau | 256 | 0.979 | 0.996 | 0.984 |
| Frans Hals | 176 | 0.974 | 0.997 | 0.982 |
| Pietro Perugino | 173 | 0.968 | 0.994 | 0.978 |
| … | | | | |
| Boris Kustodiev | 633 | 0.700 | 0.827 | 0.729 |
| Salvador Dalí | 479 | 0.675 | 0.876 | 0.725 |
| Vasily Vereshchagin | 166 | 0.666 | 0.940 | 0.709 |
| Viktor Vasnetsov | 132 | 0.682 | 0.816 | 0.704 |
| M. C. Escher | 126 | 0.558 | 0.916 | 0.602 |

</div>

<sub>Mahalanobis-distance one-class classification on WikiArt (125 named artists, 80/20 split, equal genuine/impostor test sets, seed 42). Artists with distinctive visual signatures (Michelangelo, El Greco, ukiyo-e prints) approach perfect separation; stylistically versatile artists (Dalí, Escher) are harder to model as a single distribution. Full per-artist table on [Kaggle](https://www.kaggle.com/code/ladyfaye/artsleuth-forgery-validation).</sub>

</details>

<br>

---

<br>

### ✦ Related Work & Honest Limitations

Automated art classification has a rich history, and ArtSleuth builds on the shoulders of work we want to acknowledge properly.

<br>

**Prior art in style classification.** &ensp;Saleh & Elgammal (2016) were among the first to apply metric learning to large-scale art datasets. Tan et al. (2016) trained a ResNet-50 on WikiArt and reported ~54 % style accuracy; their subsequent ArtGAN work (Tan et al., 2018) improved this to ~58 % by leveraging generative training. Chu & Wu (2018) showed that Gram-matrix representations of neural style features could reach ~63 %. More recently, multi-phase patch-based strategies (Bani & Abu-Naser, 2023) have reported high accuracy, though typically on reduced class sets or with micro-averaged metrics that weight common styles more heavily.

**Backbone representations.** &ensp;Our fusion approach is motivated by the observation — articulated clearly in recent work on style disentanglement (Pang et al., 2025) — that self-supervised models like DINOv2 (Oquab et al., 2024) and vision-language models like CLIP (Radford et al., 2021) encode fundamentally different aspects of visual style. DINOv2 captures texture and structure; CLIP captures semantic-categorical associations. Cross-attention lets each backbone inform the other, but we should note that this idea is closely related to multi-modal fusion strategies explored in VQA and image-text retrieval.

**Workshop attribution.** &ensp;Computational connoisseurship traces back to Lyu et al. (2004), who applied wavelet statistics to distinguish Bruegel from his imitators, and to Johnson et al. (2008), who used canvas-thread analysis for Vermeer attribution. Our Dirichlet-process approach to workshop decomposition is more flexible than these hand-crafted pipelines but has not yet been validated on the expert-curated datasets those studies used.

<br>

<div align="center">

| Method | Style Acc | Artist Acc | Classes | Protocol |
|:---|:---:|:---:|:---:|:---|
| ResNet-50 (Tan et al., 2016) | 54.5 % | 56.5 % | 27 / 23 | WikiArt subset, weighted avg |
| ArtGAN (Tan et al., 2018) | 58.0 % | — | 27 | WikiArt, GAN-augmented |
| Gram matrices (Chu & Wu, 2018) | 63.0 % | — | 27 | WikiArt, micro avg |
| Deep ensemble (Manzoor et al., 2024) | 68.6 % | — | 27 | WikiArt, stacking ensemble |
| ArtFusionNet (Kose & Guner, 2025) | 99.0 % | — | 3 | WikiArt subset, 3 styles only |
| ArtSleuth Fusion · e2e | 72.7 % | 79.0 % | 27 / 129 | WikiArt full, 81k, macro avg |

</div>

<sub>Numbers are taken from the respective publications. Direct comparison is difficult: studies differ in the number of classes, dataset splits, averaging methods (micro vs. macro), and whether test sets overlap with training data. We list the protocol details we could verify so readers can judge for themselves.</sub>

<br>

**Where we fall short — and we know it.**

- **Compute-constrained training.** &ensp;Fine-tuning ran for 5 epochs on a single Tesla P100 within a 12-hour Kaggle session. More epochs, larger effective batches, or higher-VRAM GPUs (A100, H100) would very likely improve the numbers. We chose to report what we could reproduce on freely available hardware rather than extrapolate.

- **Frozen-fusion underperformance.** &ensp;Our frozen cross-attention fusion (65.0 % style) actually trails bare CLIP (67.1 %). The fusion head needs gradient signal from task labels to learn a useful alignment — it does not help out of the box. We report this rather than hide it.

- **No standardised benchmark protocol.** &ensp;WikiArt classification has no single accepted evaluation protocol. Class counts, splits, and averaging methods vary between papers, which makes apples-to-apples comparison frustratingly difficult. Our numbers use macro-averaging, which is the most conservative choice (each of the 27 styles counts equally, regardless of how many images it contains). Papers that report micro-averaged or weighted scores will appear higher on the same data.

- **Forgery detection validated on embeddings, not on physical forgeries.** &ensp;We validated the one-class anomaly detector (Mahalanobis distance) on WikiArt across all 125 named artists with ≥ 80 works. Mean ROC-AUC: **0.958** (CLIP), **0.891** (fused DINOv2 + CLIP), **0.867** (DINOv2 alone). Median fused AUC is 0.902; four artists reach perfect 1.000. Full per-artist results on [Kaggle](https://www.kaggle.com/code/ladyfaye/artsleuth-forgery-validation). However, this evaluates embedding-space separation between *different* artists — it does not test against actual physical forgeries authenticated by conservators, which is a harder and more practically relevant problem.

- **Workshop decomposition is unsupervised.** &ensp;The Dirichlet-process model infers "hands" from embedding clusters, but there is no ground-truth labelled dataset of workshop paintings with per-region hand annotations to validate against. Art-historical validation by domain experts is still needed.

- **Temporal drift requires dated references.** &ensp;The Gaussian-process date estimator only works for artists whose dated reference embeddings are in the registry. For lesser-documented artists, the model has nothing to condition on.

We consider these open problems, not failures. Contributions that address any of them — especially expert-curated evaluation datasets — would strengthen the project considerably.

<br>

<details>
<summary>&nbsp;Full reference list</summary>

<br>

- Bani, M. & Abu-Naser, S. S. (2023). Artistic style recognition: combining deep and shallow neural networks for painting classification. *Mathematics*, 11(22), 4564.
- Blei, D. M. & Jordan, M. I. (2006). Variational inference for Dirichlet process mixtures. *Bayesian Analysis*, 1(1), 121–143.
- Caron, M. et al. (2021). Emerging properties in self-supervised vision transformers. *ICCV*.
- Chu, W.-T. & Wu, Y.-L. (2018). Image style classification based on learnt deep correlation features. *IEEE Trans. Multimedia*, 20(9), 2491–2502.
- Johnson, C. R. et al. (2008). Image processing for artist identification. *IEEE Signal Processing Magazine*, 25(4), 37–48.
- Kose, U. & Guner, B. (2025). Enhancing artistic style classification through a novel ArtFusionNet framework. *Scientific Reports*, 15, 20087. *(Note: evaluated on 3 style classes.)*
- Lyu, S., Rockmore, D. & Farid, H. (2004). A digital technique for art authentication. *PNAS*, 101(49), 17006–17010.
- Manzoor, T. et al. (2024). Deep ensemble art style recognition. *arXiv:2405.11675*.
- Oquab, M. et al. (2024). DINOv2: Learning robust visual features without supervision. *TMLR*.
- Pang, K. et al. (2025). StyleDecoupler: generalizable artistic style disentanglement. *arXiv:2601.17697*.
- Radford, A. et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.
- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Saleh, B. & Elgammal, A. (2016). Large-scale classification of fine-art paintings. *JOCCH*, 8(4), 1–24.
- Tan, W. R. et al. (2016). Ceci n'est pas une pipe: a deep convolutional network for fine-art paintings classification. *ICIP*.
- Tan, W. R. et al. (2018). ArtGAN: artwork synthesis with conditional categorical GANs. *ICIP*.
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.

</details>

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
│   │   ├── explainability.py  #   Gradient saliency overlays
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
  title     = {{ArtSleuth}: Computational Art Analysis Framework},
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
