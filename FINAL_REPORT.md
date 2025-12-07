# Final Report: Human-AI Collaboration Friction Analyzer

## MSR Challenge 2025 - Code Review Sentiment Analysis

**Date:** December 5, 2024
**Dataset:** AIDev (Hugging Face: `hao-li/AIDev`)
**Results Directory:** `results/run_20251205_221034/`

---

## Executive Summary

This study analyzes friction in code review comments comparing AI-assisted pull requests (PRs) with human-generated PRs. Using RoBERTa-based sentiment analysis on 21,680 review comments, we identified patterns of friction across different AI coding assistants (Copilot, Devin, Claude Code, Cursor, OpenAI Codex) and categorized negative feedback into structured topics.

### Key Findings

1. **No statistically significant difference** between human and AI-generated PRs in terms of friction (p=0.362)
2. **Significant differences exist among AI agents** (Kruskal-Wallis p=0.009)
3. **Code Style issues dominate** negative feedback (56.4% of all friction)
4. **Strong association between friction categories and agent type** (Chi-square p<0.001)
5. **Friction weakly correlates with PR outcomes** (merge time, review iterations)

---

## 1. Dataset Overview

### 1.1 Comment Distribution by Agent

| Agent | Comments | Percentage |
|-------|----------|------------|
| Copilot | 14,655 | 67.6% |
| Devin | 3,802 | 17.5% |
| OpenAI_Codex | 2,082 | 9.6% |
| Cursor | 663 | 3.1% |
| Claude_Code | 450 | 2.1% |
| Human | 28 | 0.1% |
| **Total** | **21,680** | **100%** |

**Note:** The human baseline is limited (n=28) due to the AI-focused nature of the AIDev dataset. Results for human PRs should be interpreted with caution.

---

## 2. Research Questions & Results

### RQ1: Do AI-assisted PRs generate more/less friction than human PRs?

**Test:** Mann-Whitney U Test
**Result:** U = 272,966, p = 0.362

| Group | Mean Friction Score | N |
|-------|---------------------|---|
| Human | 0.170 | 28 |
| AI (all) | 0.205 | 21,652 |

**Conclusion:** No statistically significant difference. AI-generated code receives similar levels of negative feedback as human-generated code in code reviews.

---

### RQ2: Which specific topics generate the most friction?

#### 2.1 Friction Categories (Classified via Zero-Shot BART-MNLI)

| Category | Count | Percentage | Mean Friction Score |
|----------|-------|------------|---------------------|
| Code Style | 2,073 | 56.4% | 0.696 |
| Security | 719 | 19.6% | 0.690 |
| Testing | 553 | 15.0% | 0.709 |
| Logic | 206 | 5.6% | 0.689 |
| Documentation | 127 | 3.5% | 0.669 |

**Key Insight:** Code Style issues (formatting, naming conventions, linting) dominate friction across all AI agents, comprising over half of all negative comments.

#### 2.2 BERTopic Analysis - Top Discovered Topics

| Topic | Keywords | Example Issues |
|-------|----------|----------------|
| 0 | code_block, indentation, typo | Formatting errors, typos in panic messages |
| 1 | noise, remove, pure | Request to remove unnecessary code |
| 2 | section, headings, remove | Excessive documentation sections |
| 3 | copilot, fix, claude | Direct agent mentions for fixes |
| 4 | right, ugly, change | Code aesthetics concerns |
| 5 | revert, change, date | Unwanted modifications |

**Total Topics Discovered:** 169 (including outliers)

---

### RQ3: How do different AI agents compare in generating friction?

**Test:** Kruskal-Wallis H-Test
**Result:** H = 13.52, p = 0.009

| Agent | Mean Friction | Std Dev | Ranking |
|-------|---------------|---------|---------|
| Cursor | 0.252 | 0.282 | 1 (highest friction) |
| Devin | 0.248 | 0.273 | 2 |
| Claude_Code | 0.237 | 0.254 | 3 |
| OpenAI_Codex | 0.209 | 0.232 | 4 |
| Copilot | 0.191 | 0.247 | 5 |
| Human | 0.170 | 0.225 | 6 (lowest friction) |

**Key Insights:**
- **Cursor** generates the highest friction, possibly due to its integration approach
- **Copilot** performs best among AI agents, closest to human baseline
- **Devin** (autonomous agent) shows high friction, suggesting autonomous coding increases review pushback

#### 3.1 Category Distribution by Agent

| Category | Claude_Code | Copilot | Cursor | Devin | Human | OpenAI_Codex |
|----------|-------------|---------|--------|-------|-------|--------------|
| Code Style | 60 | 1,220 | 92 | 519 | 1 | 181 |
| Security | 23 | 433 | 30 | 169 | 2 | 62 |
| Testing | 6 | 411 | 17 | 83 | 0 | 36 |
| Logic | 6 | 120 | 11 | 48 | 1 | 20 |
| Documentation | 1 | 83 | 2 | 22 | 0 | 19 |

**Chi-Square Test:** χ² = 67.00, p = 5.58×10⁻⁷ (highly significant)

**Interpretation:** There is a strong association between the type of friction and the AI agent used. Different agents have different "friction profiles."

---

### RQ4: Does friction correlate with PR outcomes?

| Metric | Correlation (r) | P-value | Interpretation |
|--------|-----------------|---------|----------------|
| Time to Merge | 0.072 | 0.0003 | Weak positive |
| Review Iterations | 0.099 | 2.1×10⁻⁹ | Weak positive |

**Key Insights:**
- Higher friction weakly predicts longer merge times
- Higher friction weakly predicts more review iterations
- Effect sizes are small but statistically significant

---

## 3. Detailed Analysis

### 3.1 Code Style: The Dominant Friction Category

Code Style issues account for **56.4%** of all negative feedback. Common patterns include:

1. **Formatting Issues**
   - Indentation errors
   - Whitespace inconsistencies
   - Line length violations

2. **Naming Conventions**
   - Variable/function naming
   - Case style violations (camelCase vs snake_case)

3. **Code Hygiene**
   - Unnecessary comments
   - Dead code
   - Debug statements left in

**Recommendation:** AI agents should be trained/configured to better follow project-specific linting rules and style guides.

### 3.2 Security Concerns

Security issues represent **19.6%** of friction, including:
- Input validation gaps
- Authentication/authorization concerns
- Secret/credential exposure risks

### 3.3 Testing Gaps

Testing-related friction (**15.0%**) involves:
- Missing unit tests
- Test coverage concerns
- Assertion quality issues

---

## 4. Statistical Summary

| Test | Statistic | P-value | Significant? |
|------|-----------|---------|--------------|
| Mann-Whitney U (Human vs AI) | 272,966 | 0.362 | No |
| Kruskal-Wallis (Agent comparison) | 13.52 | 0.009 | **Yes** |
| Chi-Square (Category × Agent) | 67.00 | 5.58×10⁻⁷ | **Yes** |
| Time-to-merge correlation | r=0.072 | 0.0003 | **Yes** |
| Iterations correlation | r=0.099 | 2.1×10⁻⁹ | **Yes** |

---

## 5. Limitations

1. **Limited Human Baseline:** Only 28 human comments available for comparison (0.1% of dataset)
2. **Dataset Bias:** AIDev focuses on AI-assisted PRs; human data is sparse
3. **Sentiment Model Limitations:** RoBERTa trained on Twitter may not fully capture technical review tone
4. **Zero-Shot Classification:** Category assignments may have inherent model biases

---

## 6. Conclusions

### Main Findings

1. **AI code quality is comparable to human code** in terms of review friction (no significant difference)

2. **Significant variation exists among AI agents:**
   - Cursor and Devin generate more friction
   - Copilot and OpenAI Codex perform better
   - Agent selection matters for review efficiency

3. **Code Style is the #1 friction source:**
   - Over half of negative feedback relates to formatting/style
   - Opportunity for agents to improve linting integration

4. **Category-agent associations are significant:**
   - Different agents have distinct friction profiles
   - Teams can choose agents based on their codebase priorities

5. **Friction predicts PR outcomes:**
   - Higher friction → longer merge times
   - Higher friction → more review iterations

### Recommendations for Practitioners

1. **Configure agents with project style guides** to reduce Code Style friction
2. **Use Copilot for lower friction** if available options include multiple agents
3. **Review Devin/Cursor PRs more carefully** given higher friction patterns
4. **Focus automated checks on formatting** to reduce review burden

---

## 7. Generated Outputs

### Data Files
- `analyzed_comments.csv` - Full dataset with friction scores
- `friction_stats_by_agent.csv` - Statistics per agent
- `category_friction_stats.csv` - Statistics per category
- `category_agent_matrix.csv` - Category × Agent distribution
- `topic_info.csv` - BERTopic discovered topics
- `statistical_tests.csv` - All test results

### Visualizations
- `friction_boxplot.png` - Friction distribution by agent
- `sentiment_distribution.png` - Overall sentiment distribution
- `category_distribution_pie.png` - Category proportions
- `category_friction_boxplot.png` - Friction by category
- `category_agent_heatmap.png` - Category × Agent heatmap
- `category_proportion_by_agent.png` - Stacked bar chart
- `topic_agent_heatmap.png` - Topic × Agent distribution
- `temporal_evolution.png` - Friction over time

---

## Appendix: Methods

### Sentiment Analysis
- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Output:** negative/neutral/positive probabilities
- **Friction Score:** P(negative sentiment)

### Topic Modeling
- **Model:** BERTopic with `all-MiniLM-L6-v2` embeddings
- **Parameters:** min_topic_size=5
- **Input:** Negative comments only (friction score > 0.5)

### Category Classification
- **Model:** `facebook/bart-large-mnli` (zero-shot)
- **Categories:** Testing, Security, Code Style, Logic, Documentation
- **Fallback:** Keyword-based classification

### Statistical Tests
- Mann-Whitney U: Non-parametric comparison of two groups
- Kruskal-Wallis H: Non-parametric comparison of multiple groups
- Chi-Square: Independence test for categorical variables
- Point-biserial/Pearson: Correlation analysis

---

*Report generated by Friction Analysis Pipeline v2.0*
*GPU Accelerated: CUDA enabled*
*Total Processing: 21,680 comments analyzed*
