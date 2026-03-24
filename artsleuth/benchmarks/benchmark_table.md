| Backbone | Style | Style F1 | Artist | Artist Top-5 | Genre |
|:---|:---:|:---:|:---:|:---:|:---:|
| DINOv2 ViT-B/14 | 57.5% | 0.553 | 64.7% | 90.9% | 71.0% |
| CLIP ViT-L/14 | 67.1% | 0.656 | 74.6% | 95.9% | 75.0% |
| Fusion (frozen) | 65.0% | 0.633 | 71.0% | 94.2% | 74.2% |
| Fusion (fine-tuned)† | 71.6% | 0.703 | 77.8% | 96.2% | 75.1% |
| **Fusion (e2e heads)†** | **72.7%** | — | **79.0%** | **96.9%** | **76.6%** |

† Fine-tuned and e2e rows are from a separate training run with partial backbone unfreezing. Only the frozen linear-probe rows (top 3) are reproducible via `benchmarks/wikiart.py`.
