# Project Structure

This document describes the complete structure of the FrictionAI project, including source files, data directories, and output formats.

## Directory Tree

```
frictionAI/
├── main.py                    # Main analysis pipeline (single-file architecture)
├── senticr.py                 # SentiCR sentiment model implementation
├── pyproject.toml             # Project dependencies (uv package manager)
├── uv.lock                    # Locked dependency versions
├── .python-version            # Python version specification (3.10)
│
├── CLAUDE.md                  # Instructions for Claude Code AI assistant
├── PIPELINE.md                # Detailed pipeline documentation
├── FINAL_REPORT.md            # Scientific analysis report
├── STRUCTURE.md               # This file - project structure documentation
├── README.md                  # Project overview and quick start
├── paper.md                   # Draft paper content
├── friction_project.md        # Initial project planning notes
│
├── senticr_data/              # SentiCR model resources
│   ├── Contractions.txt       # English contractions mapping
│   ├── EmoticonLookupTable.txt# Emoticon to text mapping
│   ├── oracle.xlsx            # Training data (1600 labeled comments)
│   └── senticr_model_GBT.pkl  # Trained Gradient Boosting model
│
├── results/                   # Analysis output directory
│   └── run_YYYYMMDD_HHMMSS/   # Timestamped run folders
│       ├── SUMMARY.txt        # Quick reference summary
│       ├── data/              # CSV and pickle data files
│       ├── plots/             # Visualization PNG files
│       └── models/            # Saved model files
│
└── .venv/                     # Virtual environment (uv-managed)
```

---

## Source Files

### `main.py` - Main Pipeline

The entire analysis pipeline in a single file (~5,000 lines). Contains:

- **`FrictionAnalyzerProject` class**: Main orchestrator with all analysis methods
- **Initialization**: Configuration for sentiment models, embedding models, quality filters
- **14 analysis phases**: From data loading to visualization and export

Key sections:
```python
class FrictionAnalyzerProject:
    def __init__(self):                    # Lines 1-350: Configuration
    def load_data(self):                   # Lines 351-550: Dataset loading
    def preprocess_data(self):             # Lines 551-900: Text cleaning
    def analyze_sentiment(self):           # Lines 901-1200: Primary sentiment
    def analyze_sentiment_multimodel(self):# Lines 1201-1500: Multi-model validation
    def analyze_senticr(self):             # Lines 1501-1700: Domain-specific sentiment
    def analyze_emotions(self):            # Lines 1701-1900: Ekman emotions
    def extract_friction_topics(self):     # Lines 1901-2100: BERTopic modeling
    def classify_friction_categories(self):# Lines 2101-2300: Zero-shot classification
    def analyze_outcomes(self):            # Lines 2301-2600: Statistical tests
    def analyze_confounders(self):         # Lines 2601-2900: OLS regression
    def analyze_power(self):               # Lines 2901-3100: Power analysis
    def visualize_results(self):           # Lines 3101-4200: Plot generation
    def save_results(self):                # Lines 4201-4500: Export to CSV
    def run_full_pipeline(self):           # Lines 4501-4600: Execute all phases
```

### `senticr.py` - Domain-Specific Sentiment

Implementation of SentiCR (Ahmed et al., 2017) for code review sentiment:

- **`SentiCRModel` class**: Gradient Boosting classifier
- **Preprocessing**: URL removal, code snippet handling, negation detection
- **Feature extraction**: TF-IDF with domain-specific preprocessing
- **Training data**: 1,600 labeled code review comments from Gerrit

---

## Configuration Files

### `pyproject.toml`

Project metadata and dependencies:

```toml
[project]
name = "frictionai"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "bertopic>=0.17.3",          # Topic modeling
    "datasets>=4.4.1",           # HuggingFace datasets
    "scipy>=1.15.3",             # Statistical tests
    "scikit-posthocs>=0.9.0",    # Dunn's post-hoc test
    "statsmodels>=0.14.0",       # OLS regression, power analysis
    "transformers>=4.57.1",      # Sentiment models
    "torch>=2.9.1",              # Deep learning backend
    "sentence-transformers>=5.1.2", # Embeddings for BERTopic
    "fast-langdetect>=1.0.0",    # Language detection
]
```

### `.python-version`

Specifies Python version for uv:
```
3.10
```

---

## SentiCR Data Directory

### `senticr_data/`

Resources for the domain-specific SentiCR sentiment model:

| File | Description |
|------|-------------|
| `Contractions.txt` | Mapping of contractions to expanded forms (e.g., "isn't" → "is not") |
| `EmoticonLookupTable.txt` | Emoticon to text translation (e.g., ":)" → "happy") |
| `oracle.xlsx` | Ground truth training data with 1,600 labeled code review comments |
| `senticr_model_GBT.pkl` | Pre-trained Gradient Boosting classifier (~189 KB) |

---

## Results Directory Structure

Each pipeline run creates a timestamped folder:

```
results/run_20260205_141334/
├── SUMMARY.txt
├── data/
│   ├── [analyzed data CSVs]
│   ├── [statistical test CSVs]
│   ├── [topic modeling CSVs]
│   └── topics_by_language/
├── plots/
│   ├── [main visualizations]
│   ├── by_pr_type/
│   ├── comments_only/
│   ├── reviews_only/
│   └── emotions/
└── models/
    └── bertopic_by_language/
```

---

## Data Files Reference

### `data/` Directory

#### Core Analysis Data

| File | Description | Columns |
|------|-------------|---------|
| `analyzed_combined.csv` | All comments with friction scores (~11MB) | clean_body, agent, source, pr_type, friction_score, sentiment_label, detected_lang, is_merged, ... |
| `analyzed_comments_only.csv` | Inline PR comments only (~31MB) | Same as above, filtered to source="comment" |
| `analyzed_reviews_only.csv` | Review-level comments only (~25MB) | Same as above, filtered to source="review" |
| `full_results.pkl` | Complete results object (pickle) | Python dict with all computed statistics |

#### Agent Friction Statistics

| File | Description | Key Metrics |
|------|-------------|-------------|
| `friction_stats_by_agent.csv` | Mean friction per AI agent | agent, mean, count, std |

Example content:
```csv
agent,mean,count,std
Claude_Code,0.175,268,0.229
Copilot,0.270,5671,0.285
Cursor,0.210,636,0.250
Devin,0.264,2290,0.275
OpenAI_Codex,0.136,2152,0.214
```

#### Statistical Tests

| File | Description | Key Columns |
|------|-------------|-------------|
| `statistical_tests.csv` | Summary of all statistical tests | kruskal_wallis_*, pointbiserial_*, chi2_* |
| `pairwise_dunn_test.csv` | Dunn's post-hoc comparisons | pair, p_dunn_raw, p_bonferroni, p_holm, p_bh, cliff_delta, effect_size |
| `power_analysis_omnibus.csv` | Kruskal-Wallis power analysis | achieved_power, effect_size, required_n |
| `power_analysis_pairwise.csv` | Pairwise comparison power | pair, achieved_power, required_n_per_group |
| `power_sensitivity_analysis.csv` | Minimum detectable effect | alpha, power, min_effect_size |

#### Multi-Model Validation

| File | Description | Key Columns |
|------|-------------|-------------|
| `multimodel_comparison_summary.csv` | Per-model statistics | model, mean_friction, negative_pct |
| `multimodel_intermodel_agreement.csv` | Cohen's Kappa between models | model_pair, cohens_kappa, interpretation, agreement_pct |
| `multimodel_ensemble_summary.csv` | Ensemble (majority vote) results | metric, value |

Example inter-model agreement:
```csv
model_pair,cohens_kappa,interpretation,agreement_pct
twitter_roberta vs cardiffnlp_roberta,0.683,substantial,85.8%
twitter_roberta vs senticr,0.106,slight/poor,67.9%
```

#### Category Classification

| File | Description | Key Columns |
|------|-------------|-------------|
| `category_counts.csv` | Comment counts per friction category | category, count |
| `category_friction_stats.csv` | Friction by category (negative comments only) | category, mean, std, count |
| `category_agent_matrix.csv` | Category × Agent cross-tabulation | agent, Testing, Security, Code_Style, Logic, Documentation |

Friction categories:
- **Testing**: Missing tests, test failures, coverage concerns
- **Security**: Vulnerabilities, auth issues, data exposure
- **Code Style**: Formatting, naming conventions, linting
- **Logic**: Bug reports, algorithm errors, edge cases
- **Documentation**: Missing docs, unclear comments

#### PR Type Analysis

| File | Description | Key Columns |
|------|-------------|-------------|
| `pr_type_friction_stats.csv` | Friction by PR type | pr_type, mean, std, count |
| `pr_type_kruskal_wallis.csv` | Omnibus test for PR types | H_statistic, p_value, significant |
| `pr_type_dunn_test.csv` | Pairwise comparisons | pair, p_raw, cliff_delta |
| `agent_pr_type_cross_stats.csv` | Agent × PR Type matrix | agent, pr_type, mean_friction, count |
| `pr_type_agent_kw_tests.csv` | Do agents differ within PR types? | pr_type, H_stat, p_value |
| `agent_pr_type_kw_tests.csv` | Do PR types differ within agents? | agent, H_stat, p_value |

PR types:
- `feat` - New features
- `fix` - Bug fixes
- `docs` - Documentation updates
- `refactor` - Code restructuring
- `test` - Test additions/modifications
- `chore` - Maintenance tasks
- `style` - Code style changes
- `other` - Uncategorized

#### Emotion Analysis

| File | Description | Key Columns |
|------|-------------|-------------|
| `emotion_stats_by_agent.csv` | Ekman emotion proportions per agent | agent, anger, disgust, fear, joy, sadness, surprise, neutral |

Ekman emotion categories:
- anger, disgust, fear (negative valence)
- joy, surprise (positive valence)
- sadness (negative, lower arousal)
- neutral (no strong emotion)

#### Topic Modeling

| File | Description | Key Columns |
|------|-------------|-------------|
| `topic_info.csv` | Combined topics from all languages | Topic, Count, Name, Representative_Docs |
| `topic_agent_matrix.csv` | Topic × Agent distribution | agent, topic_0, topic_1, ... |
| `topics_by_language/topics_EN.csv` | English-specific topics | Topic, Count, Name |
| `topics_by_language/topics_EO.csv` | Esperanto topics (fallback language) | Topic, Count, Name |

#### Temporal and Outcome Data

| File | Description | Key Columns |
|------|-------------|-------------|
| `temporal_trends.csv` | Friction over time | month, mean_friction, count |

---

## Plots Directory Reference

### `plots/` - Main Visualizations

| File | Description |
|------|-------------|
| `friction_boxplot.png` | Box plots of friction by agent |
| `friction_violin.png` | Violin plots showing distribution shapes |
| `friction_distribution_histogram.png` | Histogram of overall friction scores |
| `friction_by_source.png` | Friction comparison: comments vs reviews |
| `friction_agent_by_source.png` | Agent × Source interaction |
| `sentiment_distribution.png` | Proportion of negative/neutral/positive by agent |
| `sentiment_by_source.png` | Sentiment by comment source |
| `temporal_evolution.png` | Friction trends over time |
| `friction_vs_iterations.png` | Scatter plot: friction vs PR iteration count |
| `friction_vs_timemerge.png` | Scatter plot: friction vs time to merge |
| `category_distribution_pie.png` | Pie chart of friction categories |
| `category_friction_boxplot.png` | Box plots by category |
| `category_agent_heatmap.png` | Heatmap: Category × Agent |
| `category_proportion_by_agent.png` | Stacked bar: category proportions |
| `topic_agent_heatmap.png` | Heatmap: Topic × Agent |

### `plots/by_pr_type/` - PR Type Analysis

| File | Description |
|------|-------------|
| `friction_by_pr_type.png` | Box plots of friction by PR type |
| `friction_agent_by_pr_type_grouped.png` | Grouped bar: Agent × PR Type |
| `friction_heatmap_type_agent.png` | Heatmap: PR Type × Agent friction |
| `friction_type_by_source.png` | PR Type × Source interaction |
| `negative_rate_by_pr_type.png` | Negative comment rate by PR type |
| `negative_rate_heatmap_type_agent.png` | Heatmap: negative rate by Type × Agent |
| `pr_type_distribution.png` | Bar chart of PR type counts |
| `sentiment_by_pr_type.png` | Sentiment distribution by PR type |

### `plots/emotions/` - Emotion Analysis

| File | Description |
|------|-------------|
| `emotion_distribution.png` | Overall Ekman emotion distribution |
| `emotion_by_agent_stacked.png` | Stacked bar: emotions per agent |
| `emotion_heatmap_by_agent.png` | Heatmap: Emotion × Agent |
| `emotion_ai_vs_human.png` | AI vs Human emotion comparison (if Human data available) |
| `negative_emotion_by_agent.png` | Negative emotions (anger, disgust, fear) by agent |

### `plots/comments_only/` and `plots/reviews_only/`

Subsets of main visualizations filtered by source type:

| File | Description |
|------|-------------|
| `friction_by_agent_[comments|reviews].png` | Friction by agent (filtered) |
| `sentiment_distribution_[comments|reviews].png` | Sentiment (filtered) |

---

## Models Directory Reference

### `models/bertopic_by_language/`

Saved BERTopic models for reproducibility:

| File | Size | Description |
|------|------|-------------|
| `bertopic_EN.pkl` | ~96 MB | English topic model (27 topics) |
| `bertopic_EO.pkl` | ~480 MB | Esperanto/multilingual fallback (2 topics) |

These models can be loaded to:
- Assign topics to new documents
- Visualize topic hierarchies
- Extract topic keywords

---

## Summary Report

### `SUMMARY.txt`

Quick reference summary generated after each run:

```
======================================================================
FRICTION ANALYSIS - SUMMARY REPORT
AI Agents Code Review Friction Analysis
======================================================================

Timestamp: 20260205_141334
Total Comments Analyzed: 11017
Filter: Repositories with 100+ GitHub stars

AI Agent Distribution:
  - Copilot: 5671
  - Devin: 2290
  - OpenAI_Codex: 2152
  - Cursor: 636
  - Claude_Code: 268

Friction Statistics (Mean Friction Score):
  - OpenAI_Codex: 0.1364 (n=2152)
  - Claude_Code: 0.1749 (n=268)
  - Cursor: 0.2098 (n=636)
  - Devin: 0.2641 (n=2290)
  - Copilot: 0.2699 (n=5671)

Statistical Tests:
  Kruskal-Wallis Test: H=554.99, p<0.0001 (Significant)
  Point-Biserial Correlation: r=-0.085, p<0.0001

----------------------------------------------------------------------
METHODOLOGICAL NOTES:
----------------------------------------------------------------------
1. Human data excluded (no review comments in AIDev dataset)
2. Category friction calculated on negative comments only
3. Multi-model validation uses 3-class models for valid Cohen's Kappa
======================================================================
```

---

## Usage Notes

### Running a New Analysis

```bash
# Activate environment
source .venv/bin/activate

# Run full pipeline
python main.py

# Results saved to: results/run_YYYYMMDD_HHMMSS/
```

### Loading Previous Results

```python
import pickle

# Load full results object
with open('results/run_20260205_141334/data/full_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Access analyzed dataframe
df = results['analyzed_df']

# Access statistical tests
stats = results['statistical_tests']
```

### Comparing Runs

Each run is isolated in its own timestamped directory, enabling:
- Before/after comparisons when modifying the pipeline
- Historical tracking of results
- Easy rollback to previous analyses

---

## File Size Reference

Typical sizes for a full run:

| Category | Typical Size |
|----------|--------------|
| `data/` directory | ~70 MB |
| `plots/` directory | ~6 MB |
| `models/` directory | ~575 MB |
| **Total per run** | ~650 MB |

The models directory is the largest component due to BERTopic's embedding storage.
