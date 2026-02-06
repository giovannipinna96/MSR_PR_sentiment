# Final Report: Code Review Friction in AI-Assisted Pull Requests

## MSR Challenge 2025 - Agentic AI Code Generation Analysis

**Date:** February 6, 2026
**Dataset:** AIDev (Hugging Face: `hao-li/AIDev`)
**Results Directory:** `results/run_20260206_093254/`

---

## Executive Summary

This study analyzes **code review friction** in AI-assisted pull requests (PRs), comparing different AI coding agents using the AIDev dataset. We analyzed **11,017 review comments** from **7,156 PRs** across repositories with 100+ GitHub stars, applying rigorous quality filtering (permissive licenses + meaningful human evaluation before closure).

Using a **multilingual sentiment analysis pipeline** with language-specific transformer models and **multi-model validation**, we computed friction scores (probability of negative sentiment) for each review comment. Statistical analysis employed **Kruskal-Wallis tests** with **Dunn's post-hoc comparisons**, **Cliff's Delta effect sizes**, and **OLS regression** controlling for confounding variables.

### Key Findings

1. **Highly significant differences among AI agents** (Kruskal-Wallis H=560.72, p<0.0001, η²=0.051)
2. **OpenAI Codex generates the lowest friction** (mean=0.136), while **Copilot generates the highest** (mean=0.270)
3. **Code Style issues dominate** friction categories (54.6% of negative comments)
4. **Friction negatively correlates with merge success** (r=-0.085, p<0.0001)
5. **Agent effects remain significant after controlling for confounders** (Adjusted R²=0.065)

---

## 1. Dataset & Methodology

### 1.1 Data Collection

**Source:** AIDev dataset from Hugging Face (`hao-li/AIDev`)

**Quality Filtering Applied:**

| Filter Step | PRs Before | PRs After | Removed |
|-------------|------------|-----------|---------|
| Initial (AI + Human) | 40,214 | - | - |
| Permissive licenses (MIT/Apache-2.0) | 40,214 | 26,126 | 14,088 |
| Closed PRs only | 26,126 | 24,470 | 1,656 |
| Meaningful human evaluation | 24,470 | 7,156 | 17,314 |

**Rationale for Quality Filters:**
- **Permissive licenses:** Ensures reproducibility and ethical use
- **Closed PRs:** Enables analysis of complete review cycles
- **Human evaluation:** Requires at least one non-author review OR comment submitted BEFORE PR closure

### 1.2 Final Dataset Composition

| Agent | Comments | Percentage |
|-------|----------|------------|
| Copilot | 5,671 | 51.5% |
| Devin | 2,290 | 20.8% |
| OpenAI_Codex | 2,152 | 19.5% |
| Cursor | 636 | 5.8% |
| Claude_Code | 268 | 2.4% |
| **Total** | **11,017** | **100%** |

**Note on Human Baseline:** The AIDev dataset does not include review comments for Human PRs (only PR metadata). Human data is excluded from statistical tests due to insufficient sample size.

### 1.3 Sentiment Analysis Approach

**Primary Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`

**Multilingual Support:**
| Language | Model | n Comments | Notes |
|----------|-------|------------|-------|
| EN | twitter-roberta-base-sentiment-latest | 10,662 | Primary model |
| ZH | finbert-tone-chinese | 100 | Chinese financial BERT |
| EO | roberta-base-multilingual-sentiment | 98 | Multilingual fallback |
| JA | bert-japanese-finetuned-sentiment | 16 | Native tokenization (fugashi) |
| FR | distilcamembert-base-sentiment | 11 | 5-star → 3-class mapping |
| Other | roberta-base-multilingual-sentiment | 130 | Multilingual fallback |

**Label Mapping Notes:**
- **French model**: Outputs 5-star ratings, mapped to 3-class (1-2★→negative, 3★→neutral, 4-5★→positive)
- **Japanese model**: Uses `fugashi` + `unidic-lite` for native tokenization

**Friction Score Definition:** P(negative sentiment | text)

---

## 2. Research Questions & Results

### RQ1: How does review comment sentiment manifest for Agentic-PRs?

**Finding:** Review comments for AI-assisted PRs show a **21.2% negative sentiment rate** overall.

| Metric | Comments | Reviews | Combined |
|--------|----------|---------|----------|
| Total items | 6,819 | 4,198 | 11,017 |
| Negative (%) | 25.9% | 13.6% | 21.2% |
| Mean friction | 0.282 | 0.163 | 0.237 |

**Interpretation:** Inline comments (code-level feedback) exhibit significantly higher negativity than top-level reviews (summary feedback). This suggests that specific code implementation details generate more friction than overall approach discussions.

---

### RQ2: Are there friction differences across different AI agents?

**Test:** Kruskal-Wallis H-Test (non-parametric ANOVA)

| Statistic | Value |
|-----------|-------|
| H-statistic | 560.72 |
| P-value | 4.90 × 10⁻¹²⁰ |
| Effect size (η²) | 0.051 (small) |
| Significant | **Yes** |

**Friction Statistics by Agent (ordered by mean):**

| Agent | Mean Friction | Std Dev | n | Rank |
|-------|---------------|---------|---|------|
| OpenAI_Codex | 0.136 | 0.214 | 2,152 | 1 (lowest) |
| Claude_Code | 0.175 | 0.229 | 268 | 2 |
| Cursor | 0.210 | 0.250 | 636 | 3 |
| Devin | 0.264 | 0.275 | 2,290 | 4 |
| Copilot | 0.270 | 0.285 | 5,671 | 5 (highest) |

#### Pairwise Comparisons (Dunn's Test with Multiple Corrections)

| Comparison | Cliff's δ | Effect | p (raw) | p (Bonf) | p (Holm) | p (BH) |
|------------|-----------|--------|---------|----------|----------|--------|
| Copilot vs OpenAI_Codex | 0.336 | **medium** | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| OpenAI_Codex vs Devin | -0.268 | small | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| OpenAI_Codex vs Cursor | -0.088 | negligible | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| Claude_Code vs OpenAI_Codex | 0.097 | negligible | 0.010 | 0.095 | 0.019 | 0.011 |
| Claude_Code vs Copilot | -0.235 | small | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| Claude_Code vs Devin | -0.176 | small | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| Cursor vs Devin | -0.151 | small | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| Copilot vs Devin | 0.046 | negligible | 0.0004 | 0.004 | 0.001 | 0.0004 |
| Copilot vs Cursor | 0.207 | small | <0.0001 | <0.0001 | <0.0001 | <0.0001 |

**Key Insights:**

1. **OpenAI Codex generates significantly less friction** than all other agents (medium effect vs Copilot)
2. **Copilot generates the highest friction** despite being the most used agent
3. **Claude Code performs second-best** after OpenAI Codex
4. **The difference between Copilot and Devin is negligible** despite both having high friction

**Why might OpenAI Codex perform better?**
- Trained specifically on code completion with extensive human feedback
- Earlier model with more conservative code generation
- Possibly more focused on code quality over feature completeness

**Why might Copilot have higher friction?**
- Broader usage across diverse codebases increases exposure to style mismatches
- Integration with multiple IDEs may lead to inconsistent behavior
- Higher volume (51.5% of data) may include more edge cases

---

### RQ3: Which specific topics generate the most friction?

#### 3.1 Category Distribution (Zero-Shot Classification)

| Category | Count | Percentage | Mean Friction |
|----------|-------|------------|---------------|
| Code Style | 1,276 | 54.6% | 0.696 |
| Testing | 435 | 18.6% | 0.723 |
| Security | 398 | 17.0% | 0.693 |
| Logic | 144 | 6.2% | 0.672 |
| Documentation | 80 | 3.4% | 0.654 |

**Statistical Test (Category Differences):**
- Kruskal-Wallis H = 26.87, p < 0.0001
- Chi-square (Category × Agent) = 57.02, p < 0.0001

**Key Insight:** **Code Style issues dominate** friction (over half of all negative comments). This includes:
- Formatting violations
- Naming convention mismatches
- Linting errors
- Whitespace inconsistencies

**Testing has the highest mean friction score** (0.723), indicating that when testing-related issues arise, they generate particularly strong negative reactions.

#### 3.2 BERTopic Analysis - Discovered Topics

**English Topics (n=2,264 negative comments, 20 topics discovered):**

| Topic | Keywords | Interpretation |
|-------|----------|----------------|
| 0 | remove, gensx, summary code | Request to remove generated code |
| 1 | pull request, review, wasn able | Review process friction |
| 2 | error instead, error, throw | Error handling concerns |
| 3 | benchmark, op, code, ns | Performance benchmarking issues |
| 4 | benchmark, github, alert | CI/CD automation concerns |

**Multilingual Topics:**
- EO (Esperanto-detected, likely code): 3 topics (benchmark-related)
- Other languages: <5 negative comments (skipped)

---

### RQ4: How do friction metrics correlate with outcomes?

#### 4.1 Merge Success Correlation

| Metric | Value | P-value | Interpretation |
|--------|-------|---------|----------------|
| Point-Biserial r | -0.085 | <0.0001 | Small negative correlation |

**Merge Rates by Agent:**

| Agent | Merge Rate | n |
|-------|------------|---|
| OpenAI_Codex | 84.5% | 2,152 |
| Claude_Code | 82.1% | 268 |
| Copilot | 78.5% | 5,671 |
| Cursor | 74.4% | 636 |
| Devin | 74.0% | 2,290 |

**Key Finding:** Higher friction → lower merge probability. Agents with lower friction (OpenAI Codex) have higher merge rates.

#### 4.2 Time-to-Merge Correlation

| Metric | Spearman r | P-value |
|--------|------------|---------|
| Friction vs Time-to-Merge | 0.145 | <0.0001 |

**Interpretation:** Higher friction weakly predicts longer merge times.

#### 4.3 Review Iterations Correlation

| Metric | Spearman r | P-value |
|--------|------------|---------|
| Friction vs Iterations | 0.409 | <0.0001 |

**Key Finding:** **Moderate correlation** between friction and review iterations. High friction PRs require more review rounds before resolution.

---

## 3. Confounding Variable Analysis

### 3.1 Potential Confounders

1. **PR Type:** fix/feat/docs may generate different reviewer tones
2. **Source Type:** inline comments vs top-level reviews
3. **Text Length:** longer comments may contain more criticism

### 3.2 OLS Regression Results

**Model 1: Unadjusted (Agents Only)**
- R² = 0.038
- Adjusted R² = 0.038

**Model 2: Adjusted (Agents + Confounders)**
- R² = 0.066
- Adjusted R² = 0.065
- Sample size: n = 11,017

**Agent Coefficients (Reference: OpenAI_Codex):**

| Agent | β (Adjusted) | 95% CI | p-value |
|-------|--------------|--------|---------|
| Claude_Code | 0.029 | [-0.005, 0.063] | 0.091 |
| Copilot | 0.096*** | [0.081, 0.111] | <0.001 |
| Cursor | 0.085*** | [0.061, 0.108] | <0.001 |
| Devin | 0.107*** | [0.091, 0.123] | <0.001 |

**Key Insights:**

1. **Claude_Code becomes non-significant** after adjustment (p=0.091), suggesting its friction difference from OpenAI_Codex is partially explained by PR type and source composition
2. **Copilot, Cursor, Devin remain highly significant** (p<0.001) after adjustment
3. **Agent coefficients remain stable** after controlling for confounders, indicating robust agent effects

---

## 4. Multi-Model Validation

### 4.1 Models Used

| Model | Description | Negative Rate |
|-------|-------------|---------------|
| twitter_roberta | Primary (124M tweets) | 20.9% |
| cardiffnlp_roberta | Validation (RoBERTa base) | 20.0% |
| SentiCR | Code review-specific | 12.5% |

### 4.2 Inter-Model Agreement (Cohen's Kappa)

| Comparison | Kappa | Agreement | Interpretation |
|------------|-------|-----------|----------------|
| twitter_roberta vs cardiffnlp_roberta | 0.683 | 85.8% | **substantial** |
| twitter_roberta vs SentiCR | 0.106 | 67.9% | slight/poor |
| cardiffnlp_roberta vs SentiCR | 0.114 | 69.2% | slight/poor |

**Mean Inter-Model κ:** 0.301 (weighted by primary RoBERTa models showing substantial agreement)

**Interpretation:**
- The two RoBERTa models show **substantial agreement** (κ=0.683)
- SentiCR (trained on code review) disagrees with general-purpose models
- This is expected: SentiCR was trained on a smaller, domain-specific dataset

### 4.3 Ensemble Results (Majority Voting)

| Metric | Value |
|--------|-------|
| Negative | 17.4% |
| Neutral | 76.1% |
| Positive | 6.5% |
| Mean confidence | 87.1% |
| Unanimous agreement | 61.7% |

---

## 5. Statistical Power Analysis

### 5.1 Omnibus Test (Kruskal-Wallis)

| Metric | Value |
|--------|-------|
| Effect size (η²) | 0.051 |
| Cohen's f | 0.231 |
| Post-hoc power | **1.000** |

**Conclusion:** Excellent power for detecting overall agent differences.

### 5.2 Pairwise Power

All pairwise comparisons achieved power > 0.85 except:
- Claude_Code vs Cursor: power = 0.054 (similar friction, negligible effect)

### 5.3 Sensitivity Analysis

Minimum detectable effect size with power=0.80:
- Cohen's d = 0.11 (small effect detectable)

---

## 6. Limitations

### 6.1 Human Baseline Unavailable

The AIDev dataset does not include review comments for Human PRs, only PR metadata. We cannot directly compare AI-generated code friction with human-generated code friction.

### 6.2 Sentiment Model Limitations

- Twitter-based models may not fully capture technical review tone
- Code snippets in comments are masked ([CODE_BLOCK]), potentially losing context
- Some language-specific models require label mapping (e.g., French 5-star → 3-class)
- Low-resource languages fall back to multilingual model with potential accuracy loss

### 6.3 Category Classification

- Zero-shot BART-MNLI classification may have inherent biases
- Categories are assigned only to negative comments (by design)

### 6.4 Confounding Variables

While we controlled for PR type, source, and text length, other confounders may exist:
- Repository coding standards
- Reviewer expertise level
- PR complexity (files changed, lines added/deleted)

---

## 7. Conclusions & Implications

### 7.1 Main Findings

1. **AI agents differ significantly in friction generation** (p<0.0001, η²=0.050)
   - OpenAI Codex: lowest friction (0.136)
   - Copilot: highest friction (0.270)
   - Medium effect size between best and worst performers

2. **Code Style is the dominant friction source** (54.7%)
   - Agents should better integrate project-specific linting rules
   - Pre-submission style checks could reduce review burden

3. **Friction predicts PR outcomes**
   - Higher friction → lower merge probability (r=-0.085)
   - Higher friction → more review iterations (r=0.409)

4. **Agent effects are robust to confounding**
   - Most differences remain significant after controlling for PR type, source, text length
   - 16-29% of friction can be attributed to confounders

### 7.2 Recommendations for Practitioners

1. **Choose agents based on friction profiles:**
   - For lower review friction: prefer OpenAI Codex or Claude Code
   - Be prepared for more iterations with Copilot/Devin PRs

2. **Implement pre-commit style enforcement:**
   - 54.7% of friction is Code Style-related
   - Automated linting before PR submission would reduce review burden

3. **Allocate more review time for high-friction agents:**
   - Devin (autonomous) and Copilot PRs may require more careful review

4. **Consider context in agent selection:**
   - Security-critical projects may benefit from agents with lower Security friction
   - Test-heavy projects should evaluate Testing category friction

### 7.3 Future Work

1. Longitudinal analysis of friction trends as agents improve
2. Repository-level friction patterns and coding culture effects
3. Reviewer experience impact on friction perception
4. Intervention studies: effect of pre-commit checks on friction reduction

---

## Appendix A: Statistical Methods

### A.1 Kruskal-Wallis Test

Non-parametric alternative to one-way ANOVA for comparing multiple groups. Used because friction scores are not normally distributed.

### A.2 Dunn's Test

Proper post-hoc test after Kruskal-Wallis. Uses the same ranking as the omnibus test (unlike Mann-Whitney U which re-ranks).

### A.3 Multiple Comparison Corrections

| Method | Controls | Use Case |
|--------|----------|----------|
| Bonferroni | FWER | Confirmatory (very conservative) |
| Holm | FWER | Recommended default |
| Benjamini-Hochberg | FDR | Exploratory |

### A.4 Cliff's Delta

Non-parametric effect size measure. Preferred over Cohen's d for non-normal distributions.

| |δ| | Interpretation |
|-----|----------------|
| < 0.147 | negligible |
| 0.147 - 0.33 | small |
| 0.33 - 0.474 | medium |
| ≥ 0.474 | large |

---

## Appendix B: Generated Outputs

### Data Files

| File | Description |
|------|-------------|
| `analyzed_combined.csv` | Full dataset (11,017 rows) |
| `friction_stats_by_agent.csv` | Mean, std, n per agent |
| `pairwise_dunn_test.csv` | All pairwise comparisons |
| `statistical_tests.csv` | Summary of all tests |
| `power_analysis_*.csv` | Power analysis results |
| `multimodel_*.csv` | Multi-model validation |

### Visualizations

| File | Description |
|------|-------------|
| `friction_boxplot.png` | Friction by agent |
| `friction_violin.png` | Distribution shapes |
| `category_*.png` | Category analysis |
| `emotions/*.png` | Emotion analysis |
| `by_pr_type/*.png` | PR type breakdown |

---

## References

1. Ahmed, T., Bosu, A., Iqbal, A., & Rahimi, S. (2017). SentiCR: A customized sentiment analysis tool for code review interactions. *ASE 2017*.

2. Dunn, O.J. (1964). Multiple comparisons using rank sums. *Technometrics*, 6(3), 241-252.

3. Romano, J., et al. (2006). Appropriate statistics for ordinal level data. *FLAIR*.

4. Landis, J.R., & Koch, G.G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159-174.

5. Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.).

---

*Report generated by Friction Analysis Pipeline v3.0*
*GPU Accelerated: CUDA enabled*
*Total Processing: 11,017 comments from 7,156 PRs*
*Last updated: February 6, 2026*
