# Pipeline Documentation: FrictionAI Analysis

## Overview

This document provides a detailed step-by-step explanation of the entire analysis pipeline implemented in `main.py`. The pipeline analyzes code review friction in AI-assisted pull requests using the AIDev dataset.

---

## Table of Contents

1. [Initialization](#1-initialization)
2. [Phase 0: Dataset Schema Inspection](#2-phase-0-dataset-schema-inspection)
3. [Phase 1: Data Loading](#3-phase-1-data-loading)
4. [Phase 1b: Preprocessing & Filtering](#4-phase-1b-preprocessing--filtering)
5. [Phase 2: Multilingual Sentiment Analysis](#5-phase-2-multilingual-sentiment-analysis)
6. [Phase 2a: Multi-Model Validation](#6-phase-2a-multi-model-validation)
7. [Phase 2b: Emotion Analysis](#7-phase-2b-emotion-analysis)
8. [Phase 3: Topic Modeling (BERTopic)](#8-phase-3-topic-modeling-bertopic)
9. [Phase 3b: Friction Category Classification](#9-phase-3b-friction-category-classification)
10. [Phase 4: Statistical Analysis](#10-phase-4-statistical-analysis)
11. [Phase 4b: Confounding Variable Analysis](#11-phase-4b-confounding-variable-analysis)
12. [Phase 4c: Power Analysis](#12-phase-4c-power-analysis)
13. [Phase 5: Visualization](#13-phase-5-visualization)
14. [Phase 6: Results Saving](#14-phase-6-results-saving)

---

## 1. Initialization

### Classe `FrictionAnalyzerProject.__init__()`

```python
class FrictionAnalyzerProject:
    def __init__(self):
```

**Operazioni eseguite:**

1. **Rilevamento dispositivo GPU/CPU:**
   - Verifica se CUDA è disponibile
   - Imposta `self.device` su "cuda" o "cpu"
   - Stampa informazioni sulla GPU se disponibile

2. **Configurazione modelli primari:**
   ```python
   self.models = {
       "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
       "topic_embedding": "all-MiniLM-L6-v2",
       "category_classifier": "facebook/bart-large-mnli"
   }
   ```

3. **Modelli di sentiment per lingua:**
   - **EN**: `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa 124M tweets)
   - **FR**: `cmarkea/distilcamembert-base-sentiment` (DistilCamemBERT, 5-star → 3-class mapping)
   - **DE**: `oliverguhr/german-sentiment-bert` (German BERT 1.8M samples)
   - **ES**: `finiteautomata/beto-sentiment-analysis` (BETO TASS 2020)
   - **IT**: `neuraly/bert-base-italian-cased-sentiment` (Italian BERT)
   - **PT**: `lucas-leme/FinBERT-PT-BR` (FinBERT Brazilian Portuguese)
   - **ZH**: `yiyanghkust/finbert-tone-chinese` (FinBERT Chinese)
   - **JA**: `koheiduck/bert-japanese-finetuned-sentiment` (Japanese BERT, requires fugashi)
   - **KO**: `WhitePeak/bert-base-cased-Korean-sentiment` (Korean BERT)
   - **RU**: `blanchefort/rubert-base-cased-sentiment` (RuBERT)
   - **MULTILINGUAL**: `clapAI/roberta-base-multilingual-sentiment` (fallback 16+ lingue)

   **Note sui modelli:**
   - **FR (French)**: Il modello output 5 classi (1-5 stars) che vengono mappate a 3 classi:
     - 1-2 stars → negative
     - 3 stars → neutral
     - 4-5 stars → positive
   - **JA (Japanese)**: Richiede `fugashi` e `unidic-lite` per la tokenizzazione nativa

4. **Modelli di embedding per BERTopic:**
   - **EN**: `all-MiniLM-L6-v2` (384 dim)
   - **ZH**: `shibing624/text2vec-base-chinese` (768 dim)
   - **JA**: `pkshatech/GLuCoSE-base-ja` (768 dim)
   - **KO**: `snunlp/KR-SBERT-V40K-klueNLI-augSTS` (768 dim)
   - **FR**: `dangvantuan/sentence-camembert-large` (1024 dim)
   - **DE**: `T-Systems-onsite/cross-en-de-roberta-sentence-transformer` (768 dim)
   - **MULTILINGUAL**: `paraphrase-multilingual-MiniLM-L12-v2` (384 dim, fallback)

5. **Categorie di friction:**
   ```python
   self.friction_categories = {
       "Testing": "software testing, test coverage, unit tests...",
       "Security": "security vulnerabilities, authentication...",
       "Code Style": "code formatting, naming conventions...",
       "Logic": "code logic, bugs, edge cases...",
       "Documentation": "code documentation, comments..."
   }
   ```

6. **Creazione directory output:**
   ```
   results/run_YYYYMMDD_HHMMSS/
   ├── plots/
   ├── data/
   └── models/
   ```

---

## 2. Phase 0: Dataset Schema Inspection

### Metodo `inspect_dataset_schema()`

**Scopo:** Verifica la struttura del dataset AIDev prima di procedere con l'analisi.

**Operazioni:**
1. Carica un sample della tabella `pull_request` in streaming
2. Mostra le colonne disponibili e valori di esempio
3. Carica un sample della tabella `pr_review_comments_v2`
4. Verifica la struttura dei dati

**Output atteso:**
```
Columns: ['id', 'number', 'title', 'body', 'agent', 'user_id', 'user',
          'state', 'created_at', 'closed_at', 'merged_at', 'repo_id',
          'repo_url', 'html_url']
```

---

## 3. Phase 1: Data Loading

### Metodo `load_data()`

**Scopo:** Scarica il dataset AIDev da Hugging Face e carica tutte le tabelle necessarie.

**Tabelle caricate:**

| Tabella | Nome HuggingFace | Descrizione |
|---------|------------------|-------------|
| PRs | `pull_request` | Metadata AI PRs (pre-filtrato 100+ stars) |
| Comments | `pr_review_comments_v2` | Commenti inline sul codice |
| Reviews | `pr_reviews` | Review top-level |
| Task Types | `pr_task_type` | Tipo PR (fix, feat, docs...) |
| Human PRs | `human_pull_request` | PRs umane baseline |
| Human Task Types | `human_pr_task_type` | Tipo PR umane |
| Repositories | `repository` | Metadata repository (per filtro licenze) |
| PR Comments | `pr_comments` | Commenti generali (per filtro interazioni) |
| PR Commits | `pr_commits` | Commits per PR |
| PR Commit Details | `pr_commit_details` | Dettagli commit |

**Validazione:**
- Verifica che i dataframe non siano vuoti
- Stampa conteggi per ogni tabella

---

## 4. Phase 1b: Preprocessing & Filtering

### Metodo `preprocess_data()`

**Scopo:** Pulisce e filtra i dati applicando criteri di qualità rigorosi.

### Step 0: Concatenazione AI + Human PRs

```python
df_prs = pd.concat([df_prs, df_human_prs], ignore_index=True)
```

Unisce le PRs AI con le PRs umane come baseline.

### Step 0b: Quality Filtering

#### Filtro 1: Licenze Permissive
```python
allowed_licenses = ['MIT', 'Apache-2.0']
df_prs = df_prs[df_prs['repo_id'].isin(allowed_repo_ids)]
```
- Mantiene solo PRs da repository con licenza MIT o Apache-2.0
- **Razionale:** Garantisce riproducibilità e uso etico dei dati

#### Filtro 2: Meaningful Human Evaluation
```python
# PRs chiuse con almeno una review O commento da non-autore PRIMA della chiusura
valid_reviews = reviews_merged[
    (reviews_merged['user'] != reviews_merged['pr_user']) &
    (reviews_merged['submitted_at'] < reviews_merged['pr_closed_at'])
]
```
- **Razionale:** Esclude PRs auto-merged o chiuse senza revisione

**Risultato atteso:** ~7,156 PRs dopo i filtri (da ~40,214 iniziali)

### Step 1: Filtro PRs < 1 minuto

```python
df_prs = df_prs[(df_prs['pr_duration_seconds'].isna()) |
                (df_prs['pr_duration_seconds'] >= 60)]
```
- Rimuove PRs chiuse in meno di 60 secondi
- **Razionale:** Probabilmente spam o test

### Step 2: Aggiunta PR Task Types

```python
df_prs = pd.merge(df_prs, df_task_types[['id', 'type', 'confidence']],
                   on='id', how='left')
df_prs['pr_type'] = df_prs['type'].fillna('unknown')
```
- Classifica PRs in: feat, fix, docs, refactor, chore, test, build, ci, perf, style

### Step 3: Estrazione Repository Info

Estrae `repo_path` e `pr_number` dagli URL per il join:
```python
def extract_pr_info(url):
    # Da "https://api.github.com/repos/owner/repo/pulls/123"
    # Estrae: ("owner/repo", "123")
```

### Step 4-5: Merge Comments/Reviews con PRs

```python
merged_comments = pd.merge(df_comments, df_prs,
    left_on=['repo_path', 'pr_number'],
    right_on=['repo_path', 'number'],
    how='inner', suffixes=('_comment', '_pr'))
merged_comments['source'] = 'comment'
```

### Step 6-7: Filtro Bot

```python
bot_patterns = [r'\[bot\]', r'jenkins', r'ci/cd', r'linter',
                r'coverage', r'dependabot', r'coderabbit', r'copilot']
merged_comments['is_bot'] = merged_comments[user_col].apply(
    lambda x: any(re.search(p, str(x).lower()) for p in bot_patterns))
merged_comments = merged_comments[~merged_comments['is_bot']]
```

### Step 8: Text Cleaning

```python
def clean_text(text):
    # Rimuove blocchi di codice: ```...``` → [CODE_BLOCK]
    text = re.sub(r'```[\s\S]*?```', '[CODE_BLOCK]', text)
    # Rimuove inline code: `...` → [CODE]
    text = re.sub(r'`[^`]+`', '[CODE]', text)
    # Normalizza whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

**Filtri aggiuntivi:**
- Rimuove commenti con < 10 caratteri
- Rimuove commenti che contengono solo codice (`[CODE_BLOCK]` o `[CODE]`)

### Step 8b: Language Detection

```python
from fast_langdetect import detect_language

def detect_lang_safe(text):
    clean_for_lang = text.replace('[CODE_BLOCK]', '').replace('[CODE]', '')
    return detect_language(clean_for_lang)
```

- Usa `fast_langdetect` per identificare la lingua
- Fallback a "UNKNOWN" per testi troppo corti o errori

### Step 9: Standardizzazione e Combinazione

```python
common_cols = ['clean_body', 'agent', 'source', 'pr_type',
               'created_at', 'closed_at', 'state', 'detected_lang', 'merged_at']

combined = pd.concat([
    merged_comments[available_cols_comments],
    merged_reviews[available_cols_reviews]
], ignore_index=True)
```

---

## 5. Phase 2: Multilingual Sentiment Analysis

### Metodo `analyze_sentiment()`

**Scopo:** Calcola il friction score usando sentiment analysis multilingue.

### Lazy Loading dei Modelli

```python
def get_or_load_pipeline(lang):
    if lang in self.sentiment_models_by_lang:
        config = self.sentiment_models_by_lang[lang]
    else:
        config = self.sentiment_models_by_lang["MULTILINGUAL"]

    loaded_pipelines[lang] = pipeline(
        "sentiment-analysis",
        model=config['model_id'],
        device=self.device_id,
        top_k=None  # Restituisce probabilità per tutte le classi
    )
```

### Estrazione Friction Score

```python
def extract_sentiment_result(res, lang):
    # Per modelli 3-class standard:
    for item in res:
        mapped = label_map.get(item['label'], item['label'])
        if mapped == 'negative':
            neg_score = item['score']

    return {
        'friction_score': neg_score,      # P(negative)
        'sentiment_label': best_label,     # negative/neutral/positive
        'model_used': lang
    }
```

**Friction Score = P(sentiment negativo)**

### Batch Processing

```python
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    preds = pipe(batch, truncation=True, max_length=max_len)
```

- `batch_size = 32`
- `max_length = 512` per EN, `256` per altri

### Output

Per ogni commento/review:
- `friction_score`: float [0, 1]
- `sentiment_label`: "negative", "neutral", "positive"
- `sentiment_model_used`: lingua del modello usato
- `is_negative`: boolean (True se label == "negative")

---

## 6. Phase 2a: Multi-Model Validation

### Metodo `analyze_sentiment_multimodel()`

**Scopo:** Verifica la robustezza dei risultati usando più modelli.

### Modelli di Validazione

| Modello | ID | Descrizione |
|---------|-----|-------------|
| twitter_roberta | cardiffnlp/twitter-roberta-base-sentiment-latest | Primario |
| cardiffnlp_roberta | cardiffnlp/twitter-roberta-base-sentiment | Validazione |
| SentiCR | custom | Specifico per code review |

### SentiCR Integration

```python
from senticr import SentiCR, create_senticr_pipeline

senticr = SentiCR()
predictions = senticr.predict(texts)
```

- **Training data:** 1600 commenti code review (Ahmed et al., 2017)
- **Tipo:** Gradient Boosting Classifier
- **Output:** Binary (negative vs non-negative)

### Inter-Model Agreement (Cohen's Kappa)

```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(labels_model1, labels_model2)
```

**Interpretazione Kappa:**
- κ < 0.20: slight/poor
- 0.20 ≤ κ < 0.40: fair
- 0.40 ≤ κ < 0.60: moderate
- 0.60 ≤ κ < 0.80: substantial
- κ ≥ 0.80: almost perfect

### Ensemble Prediction (Majority Voting)

```python
# Per ogni testo, prende il label più votato
ensemble_label = mode([label_model1, label_model2, label_model3])
ensemble_confidence = votes_for_majority / total_models
```

---

## 7. Phase 2b: Emotion Analysis

### Metodo `analyze_emotions()`

**Scopo:** Identifica le emozioni specifiche nei commenti usando il modello Ekman.

### Modello Emozioni

**EN:** `j-hartmann/emotion-english-distilroberta-base`
- **Categorie (Ekman 7):** anger, disgust, fear, joy, sadness, surprise, neutral

**MULTILINGUAL:** `MilaNLProc/xlm-emo-t`
- **Categorie (8):** anger, anticipation, disgust, fear, joy, sadness, surprise, trust

### Output

Per ogni commento:
- `emotion_label`: emozione dominante
- `emotion_score`: confidence score
- `emotion_anger`, `emotion_disgust`, ...: score per ogni emozione

### Aggregazione per Agente

```python
emotion_stats = df.groupby('agent').agg({
    'emotion_anger': 'mean',
    'emotion_disgust': 'mean',
    # ...
})
```

---

## 8. Phase 3: Topic Modeling (BERTopic)

### Metodo `extract_friction_topics()`

**Scopo:** Identifica i topic principali nei commenti negativi usando BERTopic.

### Approccio Multilingue

```python
for lang, group_df in lang_groups:
    if len(group_df) < 5:
        continue  # Skip lingue con pochi commenti

    # Carica modello embedding specifico per lingua
    if lang in self.embedding_models_by_lang:
        model_config = self.embedding_models_by_lang[lang]
    else:
        model_config = self.embedding_models_by_lang["MULTILINGUAL"]

    embedding_model = SentenceTransformer(model_config["model_id"])
```

### Configurazione BERTopic

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(
    stop_words=stopwords,
    ngram_range=(1, 2),  # Unigrams e bigrams
    min_df=min_df,
    max_df=0.95
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    min_topic_size=min(10, len(group_df) // 10),
    nr_topics="auto"
)
```

### Calcolo min_df Dinamico

```python
if len(comments_list) < 100:
    min_df = 1
elif len(comments_list) < 500:
    min_df = 2
else:
    min_df = 3
```

### Output

- `topic_info.csv`: informazioni su tutti i topic (ID, count, name, keywords)
- `topics_by_language/topics_EN.csv`, `topics_ZH.csv`, etc.
- `bertopic_by_language/bertopic_EN.pkl`, etc.

---

## 9. Phase 3b: Friction Category Classification

### Metodo `classify_friction_categories()`

**Scopo:** Classifica i commenti negativi in categorie predefinite.

### Zero-Shot Classification

```python
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=self.device_id
)

result = classifier(
    text,
    candidate_labels=["Testing", "Security", "Code Style", "Logic", "Documentation"],
    multi_label=False
)
```

**Output:**
```python
{
    'sequence': 'Your test coverage is insufficient...',
    'labels': ['Testing', 'Code Style', 'Logic', 'Security', 'Documentation'],
    'scores': [0.72, 0.15, 0.08, 0.03, 0.02]
}
```

### Keyword Fallback

Se il classifier fallisce, usa pattern matching:
```python
category_keywords = {
    "Testing": ['test', 'coverage', 'assert', 'mock', 'unit'],
    "Security": ['security', 'auth', 'vulnerability', 'xss', 'inject'],
    "Code Style": ['style', 'format', 'lint', 'indent', 'naming'],
    "Logic": ['logic', 'bug', 'error', 'null', 'edge case'],
    "Documentation": ['doc', 'comment', 'readme', 'example', 'api']
}
```

---

## 10. Phase 4: Statistical Analysis

### Metodo `analyze_outcomes()`

**Scopo:** Risponde alle Research Questions con test statistici appropriati.

### RQ2: Differenze tra Agenti (Kruskal-Wallis + Dunn's Test)

#### Kruskal-Wallis Test

```python
from scipy import stats

groups = [df[df['agent'] == agent]['friction_score'].values
          for agent in agents]
H, p = stats.kruskal(*groups)
```

**Calcolo Effect Size (η²):**
```python
n = len(df)
k = len(agents)
eta_squared = (H - k + 1) / (n - k)
```

**Interpretazione η²:**
- η² < 0.01: negligible
- 0.01 ≤ η² < 0.06: small
- 0.06 ≤ η² < 0.14: medium
- η² ≥ 0.14: large

#### Dunn's Post-Hoc Test

```python
import scikit_posthocs as sp

dunn_results = sp.posthoc_dunn(
    df, val_col='friction_score', group_col='agent', p_adjust=None
)
```

**Correzioni Multiple:**
```python
from statsmodels.stats.multitest import multipletests

# Bonferroni (FWER, molto conservativo)
_, p_bonf, _, _ = multipletests(p_values, method='bonferroni')

# Holm (FWER, meno conservativo)
_, p_holm, _, _ = multipletests(p_values, method='holm')

# Benjamini-Hochberg (FDR, esplorativo)
_, p_bh, _, _ = multipletests(p_values, method='fdr_bh')
```

#### Cliff's Delta Effect Size

```python
def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    dominance = sum((xi > yj) - (xi < yj) for xi in x for yj in y)
    return dominance / (n1 * n2)
```

**Interpretazione (Romano et al., 2006):**
- |δ| < 0.147: negligible
- 0.147 ≤ |δ| < 0.33: small
- 0.33 ≤ |δ| < 0.474: medium
- |δ| ≥ 0.474: large

### RQ4: Correlazione Friction ↔ Merge Success

#### Point-Biserial Correlation

```python
# Determina is_merged dalla colonna merged_at
df['is_merged'] = df['merged_at'].notna().astype(int)

# Calcola correlazione
r, p = stats.pointbiserialr(df['friction_score'], df['is_merged'])
```

#### Correlazione con Time-to-Merge

```python
r_time, p_time = stats.spearmanr(df['friction_score'], df['time_to_merge_hours'])
```

#### Correlazione con Review Iterations

```python
r_iter, p_iter = stats.spearmanr(df['friction_score'], df['review_iterations'])
```

---

## 11. Phase 4b: Confounding Variable Analysis

### Metodo `analyze_confounders()`

**Scopo:** Controlla per variabili confondenti usando regressione OLS.

### Confounders Considerati

1. **PR Type:** fix/feat/docs generano toni diversi
2. **Source Type:** inline comments più critici delle review
3. **Text Length:** commenti più lunghi possono contenere più critiche

### Modello 1: Unadjusted (Solo Agenti)

```python
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Variabili dummy per agenti (reference: OpenAI_Codex)
agent_dummies = pd.get_dummies(df['agent'], drop_first=False)
agent_dummies = agent_dummies.drop(columns=['agent_OpenAI_Codex'])

X = add_constant(agent_dummies.astype(float))
y = df['friction_score'].astype(float)

model_unadjusted = OLS(y, X).fit()
```

### Modello 2: Adjusted (Agenti + Confounders)

```python
# Aggiungi confounders
X_confounders = pd.DataFrame({
    'is_comment': (df['source'] == 'comment').astype(int),
    'log_text_length': np.log1p(df['clean_body'].str.len())
})

# Dummy per pr_type
pr_type_dummies = pd.get_dummies(df['pr_type_grouped'], drop_first=True)

X_full = pd.concat([agent_dummies, X_confounders, pr_type_dummies], axis=1)
model_adjusted = OLS(y, add_constant(X_full.astype(float))).fit()
```

### Interpretazione

- **Coefficient change:** quanto cambiano i β degli agenti dopo l'aggiustamento
- Se β diminuisce: parte dell'effetto era dovuto ai confounders
- Se β rimane: l'effetto è robusto

---

## 12. Phase 4c: Power Analysis

### Metodo `analyze_power()`

**Scopo:** Verifica se i sample size sono adeguati per rilevare gli effetti.

### Power Analysis Omnibus (Kruskal-Wallis)

```python
from statsmodels.stats.power import FTestAnovaPower

power_analysis = FTestAnovaPower()

# Converti η² in f (Cohen's f)
f = np.sqrt(eta_squared / (1 - eta_squared))

power = power_analysis.solve_power(
    effect_size=f,
    nobs=total_n,
    alpha=0.05,
    k_groups=n_groups
)
```

### Power Analysis Pairwise

```python
from statsmodels.stats.power import TTestIndPower

power_analysis = TTestIndPower()

for pair in pairs:
    # Converti Cliff's delta in Cohen's d
    d = delta * np.sqrt(2)

    power = power_analysis.solve_power(
        effect_size=d,
        nobs1=n1,
        ratio=n2/n1,
        alpha=0.05
    )
```

### Sensitivity Analysis

```python
# Minimo effect size rilevabile con power=0.80
min_effect = power_analysis.solve_power(
    power=0.80,
    nobs=min_n,
    alpha=0.05
)
```

---

## 13. Phase 5: Visualization

### Metodo `visualize_results()`

**Plot generati:**

| File | Descrizione |
|------|-------------|
| `friction_boxplot.png` | Boxplot friction per agente |
| `friction_violin.png` | Violin plot friction per agente |
| `friction_distribution_histogram.png` | Istogramma distribuzione friction |
| `sentiment_distribution.png` | Distribuzione sentiment labels |
| `friction_by_source.png` | Friction per source (comment vs review) |
| `friction_agent_by_source.png` | Heatmap agente × source |
| `friction_vs_iterations.png` | Scatterplot friction vs iterations |
| `friction_vs_timemerge.png` | Scatterplot friction vs time-to-merge |
| `temporal_evolution.png` | Trend temporale friction |

### Emotion Visualizations

| File | Descrizione |
|------|-------------|
| `emotion_distribution.png` | Distribuzione emozioni globale |
| `emotion_by_agent_stacked.png` | Stacked bar per agente |
| `negative_emotion_by_agent.png` | Solo emozioni negative |
| `emotion_heatmap_by_agent.png` | Heatmap emozioni × agente |

### Category Visualizations

| File | Descrizione |
|------|-------------|
| `category_distribution_pie.png` | Pie chart categorie |
| `category_friction_boxplot.png` | Boxplot friction per categoria |
| `category_agent_heatmap.png` | Heatmap categoria × agente |

### PR Type Visualizations

| File | Descrizione |
|------|-------------|
| `friction_by_pr_type.png` | Friction per tipo PR |
| `friction_heatmap_type_agent.png` | Heatmap tipo × agente |

---

## 14. Phase 6: Results Saving

### Metodo `save_results()`

**File salvati in `results/run_YYYYMMDD_HHMMSS/data/`:**

| File | Contenuto |
|------|-----------|
| `analyzed_combined.csv` | Dataset completo con friction scores |
| `analyzed_comments_only.csv` | Solo commenti inline |
| `analyzed_reviews_only.csv` | Solo review top-level |
| `friction_stats_by_agent.csv` | Statistiche per agente |
| `pairwise_dunn_test.csv` | Risultati Dunn's test |
| `statistical_tests.csv` | Tutti i test statistici |
| `power_analysis_omnibus.csv` | Power analysis omnibus |
| `power_analysis_pairwise.csv` | Power analysis pairwise |
| `multimodel_intermodel_agreement.csv` | Cohen's κ tra modelli |
| `multimodel_comparison_summary.csv` | Confronto modelli |
| `emotion_stats_by_agent.csv` | Emozioni per agente |
| `topic_info.csv` | Informazioni topic BERTopic |
| `category_friction_stats.csv` | Statistiche per categoria |

**Modelli salvati in `results/run_YYYYMMDD_HHMMSS/models/`:**

| File | Contenuto |
|------|-----------|
| `bertopic_by_language/bertopic_EN.pkl` | Modello BERTopic inglese |
| `bertopic_by_language/bertopic_ZH.pkl` | Modello BERTopic cinese |

---

## Esecuzione Pipeline

### Main Entry Point

```python
if __name__ == "__main__":
    analyzer = FrictionAnalyzerProject()
    analyzer.run_full_pipeline()
```

### Metodo `run_full_pipeline()`

```python
def run_full_pipeline(self):
    # Phase 0: Schema Inspection
    self.inspect_dataset_schema()

    # Phase 1: Data Loading & Preprocessing
    self.load_data()
    self.preprocess_data()

    # Phase 2: Sentiment Analysis
    self.analyze_sentiment()
    self.analyze_sentiment_multimodel()
    self.analyze_emotions()

    # Phase 3: Topic & Category Analysis
    self.extract_friction_topics()
    self.classify_friction_categories()

    # Phase 4: Statistical Analysis
    self.analyze_outcomes()
    self.analyze_confounders()
    self.analyze_power()

    # Phase 5: Visualization
    self.visualize_results()

    # Phase 6: Save Results
    self.save_results()
```

---

## Dipendenze Chiave

```toml
[project.dependencies]
bertopic = ">=0.17.3"
datasets = ">=4.4.1"
scipy = ">=1.15.3"
scikit-learn = ">=1.5.0"
scikit-posthocs = ">=0.9.0"      # Dunn's test
statsmodels = ">=0.14.0"         # OLS, power analysis
transformers = ">=4.57.1"
torch = ">=2.9.1"
fast-langdetect = ">=0.2.0"      # Language detection
sentence-transformers = ">=2.2.0"
```

---

## Referenze Metodologiche

1. **Dunn's Test:** Dunn, O.J. (1964). Multiple comparisons using rank sums. *Technometrics*, 6(3), 241-252.
2. **Cliff's Delta:** Romano, J., et al. (2006). Appropriate statistics for ordinal level data. *FLAIR*.
3. **Cohen's Kappa:** Landis, J.R., & Koch, G.G. (1977). *Biometrics*, 33(1), 159-174.
4. **SentiCR:** Ahmed, T., et al. (2017). SentiCR: A customized sentiment analysis tool for code review. *ASE 2017*.
5. **BERTopic:** Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.
