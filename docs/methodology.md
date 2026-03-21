# Methodology

## Theoretical Foundations

ArtSleuth's analytical framework rests on two pillars — one art-historical, one computational — whose convergence defines the project's distinctive approach.

### The Morellian Tradition

In the 1870s, Giovanni Morelli proposed a radical departure from prevailing attribution methods. Rather than evaluating the *composition* or *subject matter* of a painting — elements under conscious artistic control — Morelli argued that attribution should focus on the incidental details: the shape of fingernails, the curl of earlobes, the rendering of drapery folds. These peripheral passages, executed habitually and without deliberation, reveal the artist's hand more reliably than any consciously composed focal passage.

Bernard Berenson refined this into a systematic method of connoisseurship, and subsequent scholars (Friedländer, Ainsworth) extended it to address workshop production — the common Renaissance practice in which a master designed a composition and supervised its execution by a team of assistants, each contributing recognisable passages.

ArtSleuth formalises the Morellian intuition as a *feature-extraction problem*: the "incidental details" that Morelli identified correspond, in computational terms, to the low-level textural and structural features that self-supervised vision transformers encode.

### Self-Supervised Vision Transformers

DINOv2 (Oquab et al., 2024) learns visual representations without any labelled data, using a self-distillation objective that encourages the model to produce consistent embeddings across different augmentations of the same image. The resulting feature space encodes:

- **Texture** — the physical surface quality of brushstrokes.
- **Directionality** — the dominant orientation of painted marks.
- **Granularity** — the spatial frequency of surface variation.

These correspond directly to the diagnostic features that connoisseurs evaluate.

### Contrastive Vision-Language Models

CLIP (Radford et al., 2021) encodes images and text in a shared embedding space. This is critical for *style classification*, where categories like "Baroque" or "Impressionism" are culturally constructed labels — not purely visual properties. CLIP's linguistic grounding captures the semantic associations that define these categories.

## Pipeline

### 1. Preprocessing

Artwork images undergo three corrective transforms:

- **Varnish correction**: Attenuates the warm amber shift introduced by aged surface coatings, approximating the painting's original colour temperature.
- **Craquelure suppression**: Selective median filtering that reduces age-induced crack patterns without blurring brushstroke edges.
- **Canvas texture normalisation**: Frequency-domain filtering that attenuates the periodic weave pattern of the canvas support.

### 2. Patch Extraction

The painting is divided into local analysis patches using one of three strategies:

- **Grid**: Uniform tiling with configurable overlap.
- **Salient**: Focus on high-gradient-energy regions where brushwork is most expressive.
- **Adaptive** (default): Grid tiling augmented with saliency-weighted oversampling.

### 3. Brushstroke Analysis

For each patch, the 2×2 structure tensor is computed from the greyscale gradient field:

$$J = \begin{pmatrix} \langle g_x^2 \rangle & \langle g_x g_y \rangle \\ \langle g_x g_y \rangle & \langle g_y^2 \rangle \end{pmatrix}$$

The eigenvalues $\lambda_1 \geq \lambda_2$ yield:

- **Orientation**: $\theta = \frac{1}{2} \arctan\left(\frac{2 J_{xy}}{J_{xx} - J_{yy}}\right)$
- **Coherence**: $\frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2}$
- **Energy**: $\sqrt{\lambda_1 + \lambda_2}$

Patch embeddings from DINOv2 are clustered (k-means) to reveal regions of stylistically homogeneous brushwork, potentially corresponding to distinct hands in a workshop painting.

### 4. Style Classification

The full image is encoded via CLIP, and the resulting embedding is projected through three independent linear heads (period, school, technique). The heads are trained on labelled art-historical corpora and produce calibrated probability distributions over the respective taxonomies.

### 5. Attribution

The query painting's fused embedding (concatenation of CLIP style embedding and mean DINOv2 patch embedding) is compared against a reference gallery of authenticated works using temperature-scaled cosine similarity. Confidence intervals are derived via the normal approximation to the binomial proportion.

### 6. Forgery Detection

A one-class anomaly detector evaluates the Mahalanobis distance of the query embedding from the reference distribution:

$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

The distance is normalised to a 0–1 anomaly score via sigmoid transform. Per-feature z-scores identify the specific dimensions contributing to the anomaly.

### 7. Explainability

Grad-CAM heatmaps highlight the image regions most influential to each analytical verdict. The heatmap is composited over the original artwork at full resolution using the Inferno colourmap — chosen for its perceptual uniformity and warm tonality evocative of gallery lighting.

## Limitations

- **Training data bias**: Pre-trained backbones encode biases from their training corpora (predominantly Western photographic images). Performance on non-Western painting traditions may be degraded.
- **Probabilistic, not definitive**: All outputs are statistical estimates. ArtSleuth is a screening tool that informs expert judgement; it does not replace it.
- **Reference corpus dependency**: Attribution and forgery detection quality scales directly with the size and curation quality of the reference gallery.

## References

1. Morelli, G. (1890). *Italian Painters: Critical Studies of Their Works*.
2. Berenson, B. (1902). *The Study and Criticism of Italian Art*.
3. Ainsworth, M. W. (2005). From Connoisseurship to Technical Art History. *Getty Research Journal*, 159–176.
4. Wölfflin, H. (1915). *Principles of Art History*.
5. Schölkopf, B. et al. (2001). Estimating the Support of a High-Dimensional Distribution. *Neural Computation*, 13(7), 1443–1471.
6. Caron, M. et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. *ICCV*.
7. Radford, A. et al. (2021). Learning Transferable Visual Models from Natural Language Supervision. *ICML*.
8. Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*.
9. Oquab, M. et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. *TMLR*.
