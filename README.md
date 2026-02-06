# FrictionAI: Code Review Friction Analysis for AI-Assisted Pull Requests

A research project analyzing code review friction patterns in AI-assisted pull requests. This study compares sentiment and friction metrics across different AI coding agents (GitHub Copilot, Devin, Claude Code, OpenAI Codex, Cursor) using the AIDev dataset from Hugging Face.

## Research Questions

- **RQ1**: How does review comment sentiment manifest for AI-generated pull requests?
- **RQ2**: Are there significant friction differences across different AI coding agents?
- **RQ3**: Which specific topics (security, testing, code style, logic) generate the most friction?
- **RQ4**: How do friction metrics correlate with PR outcomes (merge success, iterations, time-to-merge)?

## Key Findings

Based on analysis of **11,017 review comments** from **7,156 pull requests**:

| AI Agent | Mean Friction | Sample Size |
|----------|---------------|-------------|
| OpenAI Codex | 0.136 | 2,152 |
| Claude Code | 0.175 | 268 |
| Cursor | 0.210 | 636 |
| Devin | 0.264 | 2,290 |
| Copilot | 0.270 | 5,671 |

**Statistical Significance**: Kruskal-Wallis H=560.72, p<0.0001, confirming significant differences between agents.

**Outcome Correlation**: Higher friction correlates with lower merge success (r=-0.085, p<0.0001) and more PR iterations (r=0.409, p<0.0001).

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd frictionAI

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Running the Analysis

```bash
# Execute full pipeline
python main.py
```

Results are saved to `results/run_YYYYMMDD_HHMMSS/` with:
- **SUMMARY.txt**: Quick reference summary
- **data/**: CSV files with all computed statistics
- **plots/**: Visualizations (PNG format)
- **models/**: Saved BERTopic models

## Pipeline Overview

The analysis pipeline consists of 14 phases:

1. **Data Loading**: Download AIDev dataset from Hugging Face
2. **Preprocessing**: Text cleaning, bot filtering, quality filters
3. **Primary Sentiment Analysis**: RoBERTa-based friction scoring
4. **Multi-Model Validation**: Cross-validation with 3 sentiment models
5. **SentiCR Analysis**: Domain-specific sentiment (trained on code reviews)
6. **Emotion Detection**: Ekman 7-category emotion classification
7. **Topic Modeling**: Multilingual BERTopic for friction topics
8. **Category Classification**: Zero-shot classification into friction types
9. **Statistical Analysis**: Kruskal-Wallis, Dunn's post-hoc tests
10. **Confounder Control**: OLS regression with confounding variables
11. **Power Analysis**: Statistical power and sensitivity analysis
12. **PR Type Analysis**: Breakdown by PR type (feat, fix, docs, etc.)
13. **Visualization**: Generate plots and heatmaps
14. **Export**: Save all results to CSV and pickle formats

## Quality Filters Applied

1. **Permissive Licenses**: Only MIT or Apache-2.0 licensed repositories
2. **Meaningful Human Evaluation**: PRs must have at least one review/comment from someone other than the PR creator, submitted before closure

## Documentation

- **[PIPELINE.md](PIPELINE.md)**: Detailed code walkthrough and methodology
- **[STRUCTURE.md](STRUCTURE.md)**: Project structure and output file descriptions
- **[FINAL_REPORT.md](FINAL_REPORT.md)**: Complete scientific analysis and findings
- **[CLAUDE.md](CLAUDE.md)**: Development instructions for Claude Code

## Statistical Methods

| Method | Purpose | Reference |
|--------|---------|-----------|
| Kruskal-Wallis H-test | Omnibus comparison across agents | Non-parametric ANOVA |
| Dunn's post-hoc test | Pairwise comparisons | Dunn (1964) |
| Cliff's Delta | Effect size estimation | Romano et al. (2006) |
| Cohen's Kappa | Inter-model agreement | Landis & Koch (1977) |
| OLS Regression | Confounder adjustment | Standard linear modeling |
| Point-Biserial Correlation | Friction vs merge outcome | Binary-continuous correlation |

Multiple comparison corrections: Bonferroni, Holm, Benjamini-Hochberg (FDR)

## Models Used

### Sentiment Analysis
- **Primary**: `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa, 124M tweets)
- **Validation**: `cardiffnlp/twitter-roberta-base-sentiment`, SentiCR (code review-specific)
- **Multilingual**: Language-specific models for FR, DE, ES, IT, PT, ZH, JA, KO, RU

### Multilingual Model Notes
- **French**: `distilcamembert-base-sentiment` outputs 5-star ratings, mapped to 3-class
- **Japanese**: Uses `fugashi` + `unidic-lite` for native tokenization
- **Other languages**: Fallback to `roberta-base-multilingual-sentiment`

### Other Models
- **Emotion**: `j-hartmann/emotion-english-distilroberta-base` (Ekman 7 emotions)
- **Topics**: BERTopic with language-specific embeddings
- **Categories**: `facebook/bart-large-mnli` (zero-shot classification)

## Dependencies

```toml
bertopic>=0.17.3
datasets>=4.4.1
scipy>=1.15.3
scikit-posthocs>=0.9.0
statsmodels>=0.14.0
transformers>=4.57.1
torch>=2.9.1
sentence-transformers>=5.1.2
fast-langdetect>=1.0.0
protobuf>=6.33.5
sentencepiece>=0.2.1
fugashi>=1.5.2        # Japanese tokenization
unidic-lite>=1.0.8    # Japanese dictionary
```

## Project Structure

```
frictionAI/
├── main.py              # Main analysis pipeline
├── senticr.py           # SentiCR model implementation
├── pyproject.toml       # Dependencies
├── results/             # Analysis outputs
├── senticr_data/        # SentiCR training data
└── *.md                 # Documentation files
```

See [STRUCTURE.md](STRUCTURE.md) for complete directory structure.

## Citation

If you use this work, please cite:

```bibtex
@article{frictionai2026,
  title={Code Review Friction in AI-Assisted Pull Requests: A Comparative Analysis of AI Coding Agents},
  author={[Authors]},
  journal={MSR 2026},
  year={2026}
}
```

## References

1. Ahmed, T., Bosu, A., Iqbal, A., & Rahimi, S. (2017). SentiCR: A customized sentiment analysis tool for code review interactions. *ASE 2017*.
2. Dunn, O.J. (1964). Multiple comparisons using rank sums. *Technometrics*, 6(3).
3. Romano, J., et al. (2006). Appropriate statistics for ordinal level data. *Florida AIR*.
4. Landis, J.R., & Koch, G.G. (1977). The measurement of observer agreement. *Biometrics*, 33(1).

## License

[MIT License](LICENSE)

---

**MSR Challenge 2026** - Mining Software Repositories Conference
