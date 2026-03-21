# Let's Build a Story Together

Thank you for considering a contribution to ArtSleuth. This project lives at the intersection of art history and machine learning — a space where disciplinary humility is as important as technical skill. Every contribution, whether it corrects a Wölfflin citation or optimises a CUDA kernel, strengthens the bridge between these fields.

## How to Contribute

### Reporting Issues

Open a [GitHub Issue](https://github.com/ladyFaye1998/ArtSleuth/issues) using the appropriate template. Be specific: include Python and PyTorch versions, the image you were analysing (if shareable), and the full traceback.

### Proposing Changes

1. **Fork** the repository and create a feature branch from `main`.
2. **Install** in development mode: `pip install -e ".[dev]"`
3. **Write code** that follows the existing conventions:
   - Full type hints on all public functions.
   - Docstrings in NumPy style with art-historical context where relevant.
   - Section headers with `# ---` for visual navigation.
4. **Write tests** covering new functionality.
5. **Lint and type-check**: `ruff check .` and `mypy artsleuth`
6. **Open a Pull Request** with a clear description of *what* and *why*.

### Areas Especially Welcome

| Area | What's Needed |
|------|---------------|
| **Reference corpora** | Curated, well-attributed image sets for specific artists or periods. Provenance documentation is essential. |
| **Preprocessing** | Transforms for non-Western painting traditions (ink wash, ukiyo-e, miniature painting) that the current Western-easel-focused pipeline may handle poorly. |
| **Evaluation benchmarks** | Standardised test sets with known attributions for quantitative evaluation. |
| **Documentation** | Improving methodology docs, adding tutorials, translating to other languages. |
| **Art-historical review** | Ensuring the taxonomy, terminology, and interpretive framing remain academically defensible. |

### Code Style

- **Tone**: Professional and academic. Comments explain *why*, not *what*. Art-historical references are encouraged where they illuminate a design choice.
- **Formatting**: Enforced by Ruff. Line length is 100 characters.
- **Naming**: Classes are `PascalCase`, functions and variables are `snake_case`. Prefer descriptive names over abbreviations — `coherence` over `coh`.

### Ethical Guidelines

ArtSleuth is a *screening tool*, not an oracle. Attribution and forgery detection are probabilistic assessments that should inform — never replace — expert judgement. When writing documentation or communicating results:

- Use hedging language: "consistent with", "suggests", "warrants further examination".
- Never claim definitive authentication or condemnation of an artwork.
- Respect intellectual property and cultural sensitivity in reference corpora.

## Code of Conduct

Be excellent to each other. Treat every contributor — art historian, engineer, student — with respect and curiosity. Disagreements about methodology are welcome; personal attacks are not.

---

*Every contribution is a brushstroke in the larger picture.*
