import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import pipeline
from bertopic import BERTopic
from scipy import stats
import re
from tqdm import tqdm
import warnings
import os
from datetime import datetime
import pickle
import torch

# Configurazione
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

class FrictionAnalyzerProject:
    def __init__(self):
        self.models = {
            # Utilizziamo un modello RoBERTa robusto per sentiment (alternativa Python-pure a SentiCR)
            "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "topic_embedding": "all-MiniLM-L6-v2",
            "category_classifier": "facebook/bart-large-mnli"  # Zero-shot classification
        }
        self.data = {}
        self.results = {}

        # Structured friction categories for RQ2
        self.friction_categories = {
            "Testing": "software testing, test coverage, unit tests, integration tests, test assertions, mocking",
            "Security": "security vulnerabilities, authentication, authorization, injection attacks, XSS, CSRF, secrets",
            "Code Style": "code formatting, naming conventions, coding style, linting, indentation, style guide",
            "Logic": "code logic, bugs, edge cases, algorithms, correctness, null checks, error handling",
            "Documentation": "code documentation, comments, README, docstrings, API documentation, examples"
        }

        # Setup output directory
        self.output_dir = "results"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"run_{self.timestamp}")

        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "models"), exist_ok=True)

        print(f"📁 Output directory created: {self.run_dir}")

    # ==========================================
    # PHASE 0: Dataset Schema Inspection
    # ==========================================
    def inspect_dataset_schema(self):
        """
        Ispeziona il dataset AIDev per verificare colonne e struttura
        prima di procedere con l'analisi principale.
        """
        print("=" * 70)
        print(">>> Phase 0: Dataset Schema Inspection")
        print("=" * 70)

        try:
            # Carica un sample di ogni subset per ispezionare lo schema
            print("\n1. Inspecting all_pull_request...")
            ds_pr = load_dataset("hao-li/AIDev", name="all_pull_request", split="train", streaming=True)
            sample_pr = next(iter(ds_pr))
            print(f"   Columns: {list(sample_pr.keys())}")
            print(f"   Sample keys with values:")
            for key in list(sample_pr.keys())[:10]:  # Print first 10 columns
                print(f"      {key}: {sample_pr[key]}")

            print("\n2. Inspecting pr_review_comments_v2...")
            ds_comments = load_dataset("hao-li/AIDev", name="pr_review_comments_v2", split="train", streaming=True)
            sample_comment = next(iter(ds_comments))
            print(f"   Columns: {list(sample_comment.keys())}")
            print(f"   Sample keys with values:")
            for key in list(sample_comment.keys())[:10]:
                print(f"      {key}: {sample_comment[key]}")

            print("\n3. Inspecting human_pull_request...")
            ds_human = load_dataset("hao-li/AIDev", name="human_pull_request", split="train", streaming=True)
            sample_human = next(iter(ds_human))
            print(f"   Columns: {list(sample_human.keys())}")
            print(f"   Sample keys with values:")
            for key in list(sample_human.keys())[:10]:
                print(f"      {key}: {sample_human[key]}")

            print("\n" + "=" * 70)
            print("Schema inspection complete. Proceeding with data loading...")
            print("=" * 70 + "\n")

        except Exception as e:
            print(f"⚠️  Warning during schema inspection: {e}")
            print("Continuing with pipeline...\n")

    # ==========================================
    # PHASE 1: Data Extraction & Preprocessing
    # ==========================================
    def load_data(self):
        """
        Scarica il dataset AIDev da Hugging Face e carica le tabelle necessarie.
        Ora include anche i dati umani per il confronto baseline.
        """
        print(">>> Phase 1: Loading AIDev dataset from Hugging Face...")

        try:
            # 1. Carica Metadata PR AI (per identificare Agenti AI e Outcome)
            print("Loading AI Pull Requests metadata...")
            ds_pr = load_dataset("hao-li/AIDev", name="all_pull_request", split="train")
            self.data['prs'] = ds_pr.to_pandas()
            print(f"   Loaded {len(self.data['prs'])} AI PRs")

            # 2. Carica Commenti di Review
            print("Loading Review Comments...")
            ds_comments = load_dataset("hao-li/AIDev", name="pr_review_comments_v2", split="train")
            self.data['comments'] = ds_comments.to_pandas()
            print(f"   Loaded {len(self.data['comments'])} review comments")

            # 3. Carica PR Umane per baseline confronto
            print("Loading Human Pull Requests (baseline)...")
            ds_human = load_dataset("hao-li/AIDev", name="human_pull_request", split="train")
            self.data['human_prs'] = ds_human.to_pandas()
            print(f"   Loaded {len(self.data['human_prs'])} human PRs")

            # 4. Carica commenti umani (se esiste un subset dedicato, altrimenti li estrarremo dopo)
            # Per ora assumiamo che i commenti umani siano linkati via PR IDs

            print(f"\nData Loading Complete:")
            print(f"  - AI PRs: {len(self.data['prs'])}")
            print(f"  - Human PRs: {len(self.data['human_prs'])}")
            print(f"  - Review Comments: {len(self.data['comments'])}")

            # Validazione: Verifica che i dataframe non siano vuoti
            if len(self.data['prs']) == 0 or len(self.data['comments']) == 0:
                raise ValueError("⚠️  ERROR: Loaded empty dataframes! Check dataset access.")

        except Exception as e:
            print(f"❌ Errore nel caricamento: {e}")
            print("Verificare connessione internet e accesso al dataset HuggingFace.")
            raise

    def preprocess_data(self):
        """
        Pulisce i testi, filtra i bot, e unisce i dataframe.
        Integra REALMENTE i dati umani dal subset human_pull_request.
        """
        print(">>> Phase 1b: Preprocessing & Filtering...")

        df_prs = self.data['prs'].copy()
        df_human_prs = self.data['human_prs'].copy()
        df_comments = self.data['comments'].copy()

        print(f"\nInspecting columns for join keys...")
        print(f"  AI PRs columns: {list(df_prs.columns)[:15]}...")  # Show first 15
        print(f"  Human PRs columns: {list(df_human_prs.columns)[:15]}...")
        print(f"  Comments columns: {list(df_comments.columns)[:15]}...")

        # === STEP 1: Merge AI PRs with Comments ===
        print("\n1. Merging AI PRs with review comments...")

        # The comments have 'pull_request_url' which is an API URL
        # We need to extract repo+number from both to match them
        # Example pull_request_url: "https://api.github.com/repos/owner/repo/pulls/123"

        # Extract PR identification from comments
        if 'pull_request_url' in df_comments.columns:
            # Extract repo and PR number from URL
            def extract_pr_info(url):
                if pd.isna(url) or not isinstance(url, str):
                    return None, None
                # Format: https://api.github.com/repos/OWNER/REPO/pulls/NUMBER
                parts = url.split('/')
                if len(parts) >= 7 and 'repos' in parts:
                    owner_repo = f"{parts[-4]}/{parts[-3]}"  # owner/repo
                    pr_number = parts[-1]  # PR number
                    return owner_repo, pr_number
                return None, None

            df_comments[['repo_path', 'pr_number']] = df_comments['pull_request_url'].apply(
                lambda x: pd.Series(extract_pr_info(x))
            )
            # Convert PR number to int
            df_comments['pr_number'] = pd.to_numeric(df_comments['pr_number'], errors='coerce')

        # Extract repo info from PRs
        if 'repo_url' in df_prs.columns:
            def extract_repo_path(url):
                if pd.isna(url) or not isinstance(url, str):
                    return None
                # Format: https://api.github.com/repos/OWNER/REPO
                parts = url.split('/')
                if len(parts) >= 5 and 'repos' in parts:
                    return f"{parts[-2]}/{parts[-1]}"
                return None

            df_prs['repo_path'] = df_prs['repo_url'].apply(extract_repo_path)

        # Now merge on repo_path + number (from PR) with pr_number (from comments)
        print(f"   Attempting merge on repo_path + PR number...")

        ai_merged = pd.merge(
            df_comments,
            df_prs,
            left_on=['repo_path', 'pr_number'],
            right_on=['repo_path', 'number'],
            how='inner',
            suffixes=('_comment', '_pr')
        )

        print(f"   ✓ AI merge successful: {len(ai_merged)} comment-PR pairs")

        if len(ai_merged) == 0:
            print("   ⚠️  WARNING: No matches found between comments and AI PRs!")
            print("   This might indicate a data inconsistency. Checking sample URLs...")
            print(f"   Sample comment URL: {df_comments['pull_request_url'].iloc[0] if len(df_comments) > 0 else 'N/A'}")
            print(f"   Sample PR repo_url: {df_prs['repo_url'].iloc[0] if len(df_prs) > 0 else 'N/A'}")

        # Ensure agent column exists and is labeled
        if 'agent' not in ai_merged.columns:
            print("   ⚠️  Warning: 'agent' column not found in AI PRs")
            ai_merged['agent'] = 'Unknown_AI'
        else:
            # Fill any null agents with 'Unknown_AI'
            ai_merged['agent'] = ai_merged['agent'].fillna('Unknown_AI')

        # === STEP 2: Merge Human PRs with Comments ===
        print("\n2. Merging Human PRs with review comments...")

        # Add 'Human' label to human PRs
        df_human_prs['agent'] = 'Human'

        # Extract repo path from human PRs (same logic as AI PRs)
        if 'repo_url' in df_human_prs.columns:
            df_human_prs['repo_path'] = df_human_prs['repo_url'].apply(extract_repo_path)

        # Merge using repo_path + number
        human_merged = pd.merge(
            df_comments,
            df_human_prs,
            left_on=['repo_path', 'pr_number'],
            right_on=['repo_path', 'number'],
            how='inner',
            suffixes=('_comment', '_pr')
        )

        print(f"   ✓ Human merge successful: {len(human_merged)} comment-PR pairs")

        if len(human_merged) == 0:
            print("   ⚠️  WARNING: No matches found between comments and Human PRs!")

        # === STEP 3: Combine AI and Human datasets ===
        print("\n3. Combining AI and Human datasets...")

        # Ensure both have the same columns (take intersection)
        common_cols = list(set(ai_merged.columns) & set(human_merged.columns))
        ai_merged = ai_merged[common_cols]
        human_merged = human_merged[common_cols]

        merged = pd.concat([ai_merged, human_merged], ignore_index=True)
        print(f"   ✓ Combined dataset: {len(merged)} total rows")
        print(f"   ✓ AI comments: {len(ai_merged)}, Human comments: {len(human_merged)}")

        # === STEP 4: Filter Bots ===
        print("\n4. Filtering bot comments...")

        # Look for user column (could be 'user_name', 'user', 'author', etc.)
        user_col = None
        for possible_name in ['user_name', 'user', 'author', 'login', 'user_comment', 'author_comment']:
            if possible_name in merged.columns:
                user_col = possible_name
                break

        if user_col:
            bot_patterns = [r'\[bot\]', r'jenkins', r'ci/cd', r'linter', r'coverage', r'dependabot']
            merged['is_bot'] = merged[user_col].apply(
                lambda x: any(re.search(p, str(x).lower()) for p in bot_patterns) if pd.notnull(x) else False
            )
            before_count = len(merged)
            merged = merged[~merged['is_bot']]
            print(f"   ✓ Filtered {before_count - len(merged)} bot comments")
        else:
            print(f"   ⚠️  Warning: Could not find user column for bot filtering")

        # === STEP 5: Text Cleaning ===
        print("\n5. Cleaning comment text...")

        def clean_text(text):
            if not isinstance(text, str):
                return ""
            # Remove code blocks
            text = re.sub(r'```[\s\S]*?```', '[CODE_BLOCK]', text)
            # Remove inline code
            text = re.sub(r'`[^`]+`', '[CODE]', text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        # Look for body column
        body_col = None
        for possible_name in ['body', 'body_comment', 'comment', 'text', 'content']:
            if possible_name in merged.columns:
                body_col = possible_name
                break

        if not body_col:
            raise ValueError(f"❌ ERROR: Could not find comment body column! Available: {merged.columns.tolist()}")

        merged['clean_body'] = merged[body_col].apply(clean_text)

        # Filter out empty comments
        before_count = len(merged)
        merged = merged[merged['clean_body'].str.len() > 10]  # At least 10 characters
        print(f"   ✓ Removed {before_count - len(merged)} empty/too-short comments")

        # === STEP 6: Validation ===
        print("\n6. Validating merged dataset...")

        if len(merged) == 0:
            raise ValueError("❌ ERROR: No data remained after preprocessing!")

        # Check agent distribution
        agent_counts = merged['agent'].value_counts()
        print(f"   ✓ Agent distribution:")
        for agent, count in agent_counts.items():
            print(f"      {agent}: {count} comments")

        # Check for required columns
        required_cols = ['clean_body', 'agent']
        missing_cols = [col for col in required_cols if col not in merged.columns]
        if missing_cols:
            raise ValueError(f"❌ ERROR: Missing required columns: {missing_cols}")

        self.dataset = merged
        print(f"\n✅ Preprocessing Complete. Analysis Dataset size: {len(self.dataset)} rows.")
        print(f"   Ready for sentiment analysis!")

    # ==========================================
    # PHASE 2: Sentiment Analysis (Friction)
    # ==========================================
    def analyze_sentiment(self):
        """
        Applica RoBERTa pre-trained per calcolare sentiment.
        Mapping: Label 0 (Neg), 1 (Neu), 2 (Pos).
        Friction Score = Probabilità Negativa.
        """
        print(">>> Phase 2: Running Sentiment Analysis (RoBERTa)...")
        
        device = 0 if torch.cuda.is_available() else -1
        print(f"   Using device: {'GPU' if device == 0 else 'CPU'}")
        sentiment_pipe = pipeline("sentiment-analysis", model=self.models['sentiment'], tokenizer=self.models['sentiment'], device=device, top_k=None)
        
        # PRODUCTION MODE: Process full dataset
        batch_size = 32
        texts = self.dataset['clean_body'].tolist()
        print(f"   Processing {len(texts)} comments in batches of {batch_size}...")
        
        results = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            # Troncamento necessario per BERT (max 512 tokens)
            preds = sentiment_pipe(batch, truncation=True, max_length=512)
            results.extend(preds)
            
        # Parsing results: Estraiamo score Negativo come "Friction Score"
        friction_scores = []
        sentiments = []
        
        for res in results:
            # res è una lista di dict [{'label': 'negative', 'score': 0.9}, ...]
            neg_score = next((item['score'] for item in res if item['label'] == 'negative'), 0)
            label = max(res, key=lambda x: x['score'])['label']
            friction_scores.append(neg_score)
            sentiments.append(label)
            
        # Riattacchiamo al dataframe (sul subset elaborato)
        subset = self.dataset.iloc[:len(texts)].copy()
        subset['friction_score'] = friction_scores
        subset['sentiment_label'] = sentiments
        subset['is_negative'] = subset['sentiment_label'] == 'negative'
        
        self.analyzed_df = subset
        print("Sentiment Analysis Complete.")

    # ==========================================
    # PHASE 3: Topic Modeling (BERTopic)
    # ==========================================
    def extract_friction_topics(self):
        """
        Applica BERTopic solo sui commenti negativi per identificare le cause di friction.
        RQ2: Which specific topics generate the most friction?
        """
        print(">>> Phase 3: Topic Modeling on Negative Comments...")
        
        negative_comments = self.analyzed_df[self.analyzed_df['is_negative']]['clean_body'].tolist()
        
        if len(negative_comments) < 10:
            print("Not enough negative comments for Topic Modeling.")
            return
            
        topic_model = BERTopic(embedding_model=self.models['topic_embedding'], min_topic_size=5)
        topics, probs = topic_model.fit_transform(negative_comments)
        
        # Estraiamo info sui topic
        topic_info = topic_model.get_topic_info()
        print("Top Friction Topics identified:")
        print(topic_info.head())
        
        # Salviamo mappa topic per uso futuro
        self.results['topics'] = topic_info
        self.topic_model = topic_model # Salva modello per visualizzazioni

    # ==========================================
    # PHASE 3b: Friction Category Classification
    # ==========================================
    def classify_friction_categories(self, use_zero_shot=True, batch_size=8):
        """
        Classify negative comments into predefined friction categories.
        Uses zero-shot classification with keyword fallback.

        RQ2 Enhancement: Maps friction to structured categories:
        - Testing, Security, Code Style, Logic, Documentation
        """
        print(">>> Phase 3b: Classifying Friction Categories...")

        # Filter negative comments
        negative_mask = self.analyzed_df['is_negative']
        negative_df = self.analyzed_df[negative_mask].copy()

        if len(negative_df) == 0:
            print("   No negative comments to classify.")
            self.analyzed_df['friction_category'] = 'N/A'
            return

        print(f"   Classifying {len(negative_df)} negative comments...")

        # Define category labels for zero-shot
        category_labels = list(self.friction_categories.keys())

        if use_zero_shot:
            categories = self._classify_zero_shot(
                negative_df['clean_body'].tolist(),
                category_labels,
                batch_size
            )
        else:
            categories = self._classify_keywords(negative_df['clean_body'].tolist())

        # Add category to negative comments
        negative_df['friction_category'] = categories

        # Initialize all rows with 'N/A' then update negative ones
        self.analyzed_df['friction_category'] = 'N/A'
        self.analyzed_df.loc[negative_mask, 'friction_category'] = categories

        # Store category statistics
        category_counts = pd.Series(categories).value_counts()
        self.results['category_counts'] = category_counts

        print("   Category distribution:")
        for cat, count in category_counts.items():
            pct = 100 * count / len(negative_df)
            print(f"      {cat}: {count} ({pct:.1f}%)")

    def _classify_zero_shot(self, texts, labels, batch_size=8):
        """
        Zero-shot classification using BART-MNLI.
        """
        device = 0 if torch.cuda.is_available() else -1
        print(f"   Using device for classification: {'GPU' if device == 0 else 'CPU'}")

        classifier = pipeline(
            "zero-shot-classification",
            model=self.models['category_classifier'],
            device=device
        )

        # Create hypothesis template for better accuracy
        hypothesis_template = "This code review comment is about {}."

        categories = []
        for i in tqdm(range(0, len(texts), batch_size), desc="   Classifying"):
            batch = texts[i:i+batch_size]

            for text in batch:
                # Truncate very long texts to avoid issues
                text = text[:1000] if len(text) > 1000 else text

                try:
                    result = classifier(
                        text,
                        candidate_labels=labels,
                        hypothesis_template=hypothesis_template,
                        multi_label=False
                    )
                    categories.append(result['labels'][0])  # Top predicted category
                except Exception as e:
                    # Fallback to keyword-based for problematic texts
                    categories.append(self._classify_single_keyword(text))

        return categories

    def _classify_keywords(self, texts):
        """
        Fast keyword-based classification as fallback.
        """
        return [self._classify_single_keyword(text) for text in texts]

    def _classify_single_keyword(self, text):
        """
        Classify a single text using keyword matching.
        """
        text_lower = text.lower()

        keyword_map = {
            "Testing": ["test", "coverage", "unit", "integration", "mock", "assert", "pytest", "jest", "spec"],
            "Security": ["security", "vulnerab", "auth", "inject", "xss", "csrf", "secret", "password", "token"],
            "Code Style": ["format", "naming", "convention", "lint", "style", "indent", "camel", "snake"],
            "Logic": ["bug", "edge case", "algorithm", "correct", "null", "error", "exception", "logic", "fix"],
            "Documentation": ["doc", "comment", "readme", "docstring", "explain", "example", "description"]
        }

        scores = {}
        for category, keywords in keyword_map.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[category] = score

        # Return category with highest score, or "Other" if no matches
        best_category = max(scores, key=scores.get)
        return best_category if scores[best_category] > 0 else "Other"

    # ==========================================
    # PHASE 4: Correlation & Statistical Analysis
    # ==========================================
    def analyze_outcomes(self):
        """
        RQ1, RQ3, RQ4: Aggrega metriche e calcola statistiche.
        """
        print(">>> Phase 4: Statistical Analysis...")
        df = self.analyzed_df
        
        # RQ1 & RQ3: Friction by Agent vs Human
        friction_by_agent = df.groupby('agent')['friction_score'].agg(['mean', 'count', 'std']).reset_index()
        self.results['friction_stats'] = friction_by_agent
        
        # Statistical Test (Mann-Whitney U) Human vs AI (se abbiamo entrambi)
        humans = df[df['agent'] == 'Human']['friction_score']
        ais = df[df['agent'] != 'Human']['friction_score']
        
        if len(humans) > 0 and len(ais) > 0:
            stat, p_val = stats.mannwhitneyu(humans, ais, alternative='two-sided')
            self.results['mann_whitney'] = {'stat': stat, 'p_value': p_val}
            print(f"Human vs AI Friction Difference: p-value = {p_val:.4f}")
        
        # RQ4: Correlation with Merge Outcome
        # Check if we have state column
        state_col = None
        for possible_name in ['state', 'state_pr', 'status', 'merged']:
            if possible_name in df.columns:
                state_col = possible_name
                break

        if state_col:
            # Logistic Regression proxy (Point-biserial correlation)
            df['is_merged'] = df[state_col].apply(lambda x: 1 if 'merge' in str(x).lower() else 0)
            corr, p_val_corr = stats.pointbiserialr(df['friction_score'], df['is_merged'])
            self.results['correlation'] = {'r': corr, 'p': p_val_corr}
            print(f"Correlation Friction <-> Merge Success: r = {corr:.3f}, p = {p_val_corr:.4f}")
        else:
            print("   ⚠️  Warning: No state/outcome column found for correlation analysis")

    # ==========================================
    # PHASE 4b: Category-Based Friction Analysis
    # ==========================================
    def analyze_category_friction(self):
        """
        RQ2 Enhanced: Statistical analysis of friction by category.
        Includes Kruskal-Wallis and Chi-square tests.
        """
        print(">>> Phase 4b: Category-Based Friction Analysis...")

        if 'friction_category' not in self.analyzed_df.columns:
            print("   No friction categories found. Run classify_friction_categories first.")
            return

        df = self.analyzed_df[self.analyzed_df['friction_category'] != 'N/A'].copy()

        if len(df) == 0:
            print("   No categorized comments for analysis.")
            return

        # Friction by category
        category_stats = df.groupby('friction_category').agg({
            'friction_score': ['mean', 'std', 'count']
        }).round(4)
        category_stats.columns = ['mean_friction', 'std_friction', 'count']
        category_stats = category_stats.reset_index()

        self.results['category_friction_stats'] = category_stats
        print("   Category friction statistics:")
        print(category_stats.to_string(index=False))

        # Kruskal-Wallis test across categories (non-parametric ANOVA)
        category_groups = [group['friction_score'].values for name, group in df.groupby('friction_category')]
        if len(category_groups) >= 2 and all(len(g) >= 2 for g in category_groups):
            try:
                stat, p_val = stats.kruskal(*category_groups)
                self.results['kruskal_wallis_categories'] = {'stat': stat, 'p_value': p_val}
                sig = "Yes" if p_val < 0.05 else "No"
                print(f"\n   Kruskal-Wallis test (categories): H={stat:.2f}, p={p_val:.4f}, Significant: {sig}")
            except Exception as e:
                print(f"   Kruskal-Wallis test failed: {e}")
        else:
            print("   Insufficient data for Kruskal-Wallis test")

        # Category by Agent interaction
        category_agent = df.groupby(['friction_category', 'agent']).size().unstack(fill_value=0)
        self.results['category_agent_matrix'] = category_agent

        # Chi-square test for category-agent independence
        if category_agent.shape[0] >= 2 and category_agent.shape[1] >= 2:
            try:
                chi2, p_val, dof, expected = stats.chi2_contingency(category_agent)
                self.results['chi2_category_agent'] = {'chi2': chi2, 'p_value': p_val, 'dof': dof}
                sig = "Yes" if p_val < 0.05 else "No"
                print(f"   Chi-square test (category vs agent): chi2={chi2:.2f}, p={p_val:.4f}, Significant: {sig}")
            except Exception as e:
                print(f"   Chi-square test failed: {e}")
        else:
            print("   Insufficient data for Chi-square test")

    # ==========================================
    # ENHANCED RQ5: Temporal Evolution Analysis
    # ==========================================
    def analyze_temporal_evolution(self):
        """
        RQ5: Analizza come la friction evolve nel tempo.
        Identifica se i reviewer si adattano riducendo la frustrazione.
        """
        print(">>> Enhanced RQ5: Temporal Evolution of Friction...")
        df = self.analyzed_df

        # Find timestamp column
        time_col = None
        for possible_name in ['created_at', 'created_at_comment', 'timestamp', 'date']:
            if possible_name in df.columns:
                time_col = possible_name
                break

        if not time_col:
            print("   ⚠️  Warning: No timestamp column found. Skipping temporal analysis.")
            return

        # Convert to datetime
        df['timestamp'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        if len(df) < 10:
            print("   ⚠️  Warning: Not enough timestamped comments for temporal analysis.")
            return

        # Group by month and agent
        df['year_month'] = df['timestamp'].dt.to_period('M')

        # Calculate monthly friction trends
        temporal_trends = df.groupby(['year_month', 'agent'])['friction_score'].agg(['mean', 'count']).reset_index()
        temporal_trends = temporal_trends[temporal_trends['count'] >= 5]  # At least 5 comments per period

        if len(temporal_trends) == 0:
            print("   ⚠️  Warning: Not enough data points for temporal analysis.")
            return

        self.results['temporal_trends'] = temporal_trends
        print(f"   ✓ Temporal analysis complete: {len(temporal_trends)} month-agent datapoints")

        # Visualization: Time series plot
        plt.figure(figsize=(14, 7))
        for agent in temporal_trends['agent'].unique():
            agent_data = temporal_trends[temporal_trends['agent'] == agent]
            plt.plot(
                agent_data['year_month'].astype(str),
                agent_data['mean'],
                marker='o',
                label=agent,
                linewidth=2
            )

        plt.xlabel("Time Period (Year-Month)", fontsize=12)
        plt.ylabel("Mean Friction Score", fontsize=12)
        plt.title("Temporal Evolution of Friction: Adaptation Over Time", fontsize=14, fontweight='bold')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(self.run_dir, "plots", "temporal_evolution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved temporal plot: {plot_path}")
        plt.close()

        # Statistical test: Correlation between time and friction (per agent)
        print("\n   Temporal correlation analysis (time vs friction):")
        for agent in df['agent'].unique():
            agent_df = df[df['agent'] == agent].copy()
            if len(agent_df) < 10:
                continue

            # Numeric time encoding
            agent_df['time_numeric'] = (agent_df['timestamp'] - agent_df['timestamp'].min()).dt.days

            corr, p_val = stats.spearmanr(agent_df['time_numeric'], agent_df['friction_score'])
            print(f"      {agent}: r={corr:.3f}, p={p_val:.4f} {'✓' if p_val < 0.05 else ''}")

    # ==========================================
    # ENHANCED RQ2: Topic-Agent Interaction Matrix
    # ==========================================
    def analyze_topic_agent_interaction(self):
        """
        Enhanced RQ2: Crea matrice topic × agent per identificare
        quali topic generano più friction per ciascun agente.
        """
        print(">>> Enhanced RQ2: Topic-Agent Interaction Matrix...")

        if not hasattr(self, 'topic_model'):
            print("   ⚠️  Warning: No topic model available. Run extract_friction_topics first.")
            return

        df = self.analyzed_df[self.analyzed_df['is_negative']].copy()

        if len(df) < 10:
            print("   ⚠️  Warning: Not enough negative comments for topic-agent analysis.")
            return

        # Assign topics to negative comments
        topics, _ = self.topic_model.transform(df['clean_body'].tolist())
        df['topic'] = topics

        # Remove outliers (topic -1)
        df = df[df['topic'] != -1]

        if len(df) == 0:
            print("   ⚠️  Warning: No valid topics found.")
            return

        # Create topic-agent interaction matrix
        interaction_matrix = df.groupby(['topic', 'agent']).size().unstack(fill_value=0)

        self.results['topic_agent_matrix'] = interaction_matrix
        print(f"   ✓ Created topic-agent matrix: {interaction_matrix.shape}")

        # Visualization: Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(interaction_matrix, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
        plt.title("Topic-Agent Interaction Matrix: Friction Sources by Agent", fontsize=14, fontweight='bold')
        plt.xlabel("Agent", fontsize=12)
        plt.ylabel("Topic ID", fontsize=12)
        plt.tight_layout()

        plot_path = os.path.join(self.run_dir, "plots", "topic_agent_heatmap.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved topic-agent heatmap: {plot_path}")
        plt.close()

        # Top topics per agent
        print("\n   Top friction topics by agent:")
        for agent in interaction_matrix.columns:
            top_topics = interaction_matrix[agent].nlargest(3)
            print(f"      {agent}:")
            for topic_id, count in top_topics.items():
                if count > 0:
                    topic_words = self.topic_model.get_topic(topic_id)[:3]  # Top 3 words
                    words = ', '.join([word for word, _ in topic_words])
                    print(f"         Topic {topic_id} ({words}): {count} comments")

    # ==========================================
    # ENHANCED RQ4: Time-to-Merge and Iterations
    # ==========================================
    def analyze_timemerge_iterations(self):
        """
        Enhanced RQ4: Analizza correlazione tra friction e:
        - Time-to-merge (durata della PR)
        - Numero di iterazioni (review cycles)
        """
        print(">>> Enhanced RQ4: Time-to-Merge and Review Iterations...")
        df = self.analyzed_df

        # === Time-to-Merge Analysis ===
        created_col = None
        merged_col = None

        for possible_name in ['created_at', 'created_at_pr', 'pr_created_at']:
            if possible_name in df.columns:
                created_col = possible_name
                break

        for possible_name in ['merged_at', 'merged_at_pr', 'closed_at', 'closed_at_pr']:
            if possible_name in df.columns:
                merged_col = possible_name
                break

        if created_col and merged_col:
            df['created_time'] = pd.to_datetime(df[created_col], errors='coerce')
            df['merged_time'] = pd.to_datetime(df[merged_col], errors='coerce')

            # Calculate time-to-merge in hours
            df['time_to_merge_hours'] = (df['merged_time'] - df['created_time']).dt.total_seconds() / 3600

            # Filter valid values
            valid_merge = df.dropna(subset=['time_to_merge_hours'])
            valid_merge = valid_merge[valid_merge['time_to_merge_hours'] > 0]
            valid_merge = valid_merge[valid_merge['time_to_merge_hours'] < 8760]  # Less than 1 year

            if len(valid_merge) >= 10:
                # Aggregate friction by PR
                pr_id_col = 'id_pr' if 'id_pr' in valid_merge.columns else valid_merge.columns[0]
                pr_aggregated = valid_merge.groupby(pr_id_col).agg({
                    'friction_score': 'mean',
                    'time_to_merge_hours': 'first',
                    'agent': 'first'
                }).reset_index()

                # Correlation
                corr, p_val = stats.spearmanr(pr_aggregated['friction_score'], pr_aggregated['time_to_merge_hours'])
                self.results['time_to_merge_correlation'] = {'r': corr, 'p': p_val}
                print(f"   ✓ Time-to-merge correlation: r={corr:.3f}, p={p_val:.4f}")

                # Visualization: Scatter plot
                plt.figure(figsize=(12, 7))
                for agent in pr_aggregated['agent'].unique():
                    agent_data = pr_aggregated[pr_aggregated['agent'] == agent]
                    plt.scatter(
                        agent_data['friction_score'],
                        agent_data['time_to_merge_hours'],
                        label=agent,
                        alpha=0.6,
                        s=50
                    )

                plt.xlabel("Mean Friction Score", fontsize=12)
                plt.ylabel("Time to Merge (hours)", fontsize=12)
                plt.title("Friction vs Time-to-Merge", fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                plot_path = os.path.join(self.run_dir, "plots", "friction_vs_timemerge.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"   ✓ Saved time-to-merge plot: {plot_path}")
                plt.close()
            else:
                print("   ⚠️  Warning: Not enough data for time-to-merge analysis.")
        else:
            print("   ⚠️  Warning: Missing timestamp columns for time-to-merge analysis.")

        # === Review Iterations Analysis ===
        # Count number of comments per PR as proxy for iterations
        pr_id_col = None
        for possible_name in ['id_pr', 'pull_request_id', 'pr_id']:
            if possible_name in df.columns:
                pr_id_col = possible_name
                break

        if pr_id_col:
            pr_iterations = df.groupby(pr_id_col).agg({
                'friction_score': 'mean',
                'clean_body': 'count',  # Count comments as proxy for iterations
                'agent': 'first'
            }).reset_index()
            pr_iterations.rename(columns={'clean_body': 'comment_count'}, inplace=True)

            # Correlation
            if len(pr_iterations) >= 10:
                corr, p_val = stats.spearmanr(pr_iterations['friction_score'], pr_iterations['comment_count'])
                self.results['iterations_correlation'] = {'r': corr, 'p': p_val}
                print(f"   ✓ Review iterations correlation: r={corr:.3f}, p={p_val:.4f}")

                # Visualization
                plt.figure(figsize=(12, 7))
                for agent in pr_iterations['agent'].unique():
                    agent_data = pr_iterations[pr_iterations['agent'] == agent]
                    plt.scatter(
                        agent_data['friction_score'],
                        agent_data['comment_count'],
                        label=agent,
                        alpha=0.6,
                        s=50
                    )

                plt.xlabel("Mean Friction Score", fontsize=12)
                plt.ylabel("Number of Review Comments", fontsize=12)
                plt.title("Friction vs Review Iteration Count", fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                plot_path = os.path.join(self.run_dir, "plots", "friction_vs_iterations.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"   ✓ Saved iterations plot: {plot_path}")
                plt.close()
            else:
                print("   ⚠️  Warning: Not enough PRs for iteration analysis.")
        else:
            print("   ⚠️  Warning: No PR ID column found for iteration analysis.")

    # ==========================================
    # PHASE 5: Visualization
    # ==========================================
    def visualize_results(self):
        """
        Genera e salva i grafici richiesti.
        """
        print(">>> Phase 5: Visualization...")
        df = self.analyzed_df
        plots_dir = os.path.join(self.run_dir, "plots")

        # 1. Boxplot Friction by Agent
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='agent', y='friction_score', data=df, palette="viridis")
        plt.title("Friction Score (Negative Sentiment Probability) by Agent", fontsize=14, fontweight='bold')
        plt.ylabel("Friction Level (0-1)", fontsize=12)
        plt.xlabel("Agent Type", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "friction_boxplot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {plot_path}")
        plt.close()

        # 2. Sentiment Distribution
        plt.figure(figsize=(12, 7))
        sns.countplot(x='sentiment_label', hue='agent', data=df, palette="Set2")
        plt.title("Sentiment Label Distribution: Human vs AI Agents", fontsize=14, fontweight='bold')
        plt.xlabel("Sentiment", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.legend(title="Agent", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "sentiment_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {plot_path}")
        plt.close()

        # 3. Friction Score Distribution (Histogram)
        plt.figure(figsize=(12, 7))
        for agent in df['agent'].unique():
            agent_data = df[df['agent'] == agent]['friction_score']
            plt.hist(agent_data, alpha=0.5, label=agent, bins=30)
        plt.xlabel("Friction Score", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Distribution of Friction Scores by Agent", fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "friction_distribution_histogram.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {plot_path}")
        plt.close()

        print(f"   All visualizations saved to: {plots_dir}")

    # ==========================================
    # PHASE 5b: Category Visualizations
    # ==========================================
    def visualize_categories(self):
        """
        Generate visualizations for friction categories.
        """
        print(">>> Phase 5b: Category Visualizations...")
        plots_dir = os.path.join(self.run_dir, "plots")

        if 'friction_category' not in self.analyzed_df.columns:
            print("   No friction categories found. Skipping category visualizations.")
            return

        df = self.analyzed_df[self.analyzed_df['friction_category'] != 'N/A'].copy()

        if len(df) == 0:
            print("   No categorized data for visualization.")
            return

        # 1. Category Distribution Pie Chart
        plt.figure(figsize=(10, 8))
        category_counts = df['friction_category'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title("Friction Categories Distribution", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "category_distribution_pie.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 2. Category Friction Boxplot
        plt.figure(figsize=(12, 7))
        order = df.groupby('friction_category')['friction_score'].mean().sort_values(ascending=False).index
        sns.boxplot(x='friction_category', y='friction_score', data=df, order=order, palette="husl")
        plt.title("Friction Score by Category", fontsize=14, fontweight='bold')
        plt.xlabel("Friction Category", fontsize=12)
        plt.ylabel("Friction Score", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "category_friction_boxplot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 3. Category-Agent Heatmap
        plt.figure(figsize=(12, 8))
        category_agent = df.groupby(['friction_category', 'agent']).size().unstack(fill_value=0)
        sns.heatmap(category_agent, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
        plt.title("Friction Category by Agent", fontsize=14, fontweight='bold')
        plt.xlabel("Agent", fontsize=12)
        plt.ylabel("Friction Category", fontsize=12)
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "category_agent_heatmap.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 4. Stacked Bar Chart: Category proportion by Agent
        if len(category_agent.columns) >= 1:
            plt.figure(figsize=(12, 7))
            category_agent_pct = category_agent.div(category_agent.sum(axis=0), axis=1) * 100
            category_agent_pct.T.plot(kind='bar', stacked=True, colormap='Set3', figsize=(12, 7))
            plt.title("Friction Category Proportion by Agent", fontsize=14, fontweight='bold')
            plt.xlabel("Agent", fontsize=12)
            plt.ylabel("Percentage (%)", fontsize=12)
            plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, "category_proportion_by_agent.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   Saved: {plot_path}")
            plt.close()

        print(f"   All category visualizations saved.")

    # ==========================================
    # PHASE 6: Save Results
    # ==========================================
    def save_results(self):
        """
        Salva tutti i risultati (CSV, modelli, statistiche) su disco.
        """
        print(">>> Phase 6: Saving Results...")
        data_dir = os.path.join(self.run_dir, "data")
        models_dir = os.path.join(self.run_dir, "models")

        # 1. Save analyzed comments dataset
        analyzed_path = os.path.join(data_dir, "analyzed_comments.csv")
        self.analyzed_df.to_csv(analyzed_path, index=False)
        print(f"   ✓ Saved analyzed comments: {analyzed_path}")

        # 2. Save friction statistics by agent
        if 'friction_stats' in self.results:
            stats_path = os.path.join(data_dir, "friction_stats_by_agent.csv")
            self.results['friction_stats'].to_csv(stats_path, index=False)
            print(f"   ✓ Saved friction stats: {stats_path}")

        # 3. Save topic modeling results
        if 'topics' in self.results:
            topics_path = os.path.join(data_dir, "topic_info.csv")
            self.results['topics'].to_csv(topics_path, index=False)
            print(f"   ✓ Saved topic info: {topics_path}")

        # 3b. Save temporal trends (RQ5)
        if 'temporal_trends' in self.results:
            temporal_path = os.path.join(data_dir, "temporal_trends.csv")
            self.results['temporal_trends'].to_csv(temporal_path, index=False)
            print(f"   ✓ Saved temporal trends: {temporal_path}")

        # 3c. Save topic-agent interaction matrix (Enhanced RQ2)
        if 'topic_agent_matrix' in self.results:
            matrix_path = os.path.join(data_dir, "topic_agent_matrix.csv")
            self.results['topic_agent_matrix'].to_csv(matrix_path)
            print(f"   ✓ Saved topic-agent matrix: {matrix_path}")

        # 3d. Save category friction statistics (Enhanced RQ2)
        if 'category_friction_stats' in self.results:
            cat_stats_path = os.path.join(data_dir, "category_friction_stats.csv")
            self.results['category_friction_stats'].to_csv(cat_stats_path, index=False)
            print(f"   ✓ Saved category friction stats: {cat_stats_path}")

        # 3e. Save category-agent matrix
        if 'category_agent_matrix' in self.results:
            cat_agent_path = os.path.join(data_dir, "category_agent_matrix.csv")
            self.results['category_agent_matrix'].to_csv(cat_agent_path)
            print(f"   ✓ Saved category-agent matrix: {cat_agent_path}")

        # 3f. Save category counts
        if 'category_counts' in self.results:
            cat_counts_path = os.path.join(data_dir, "category_counts.csv")
            self.results['category_counts'].to_frame('count').to_csv(cat_counts_path)
            print(f"   ✓ Saved category counts: {cat_counts_path}")

        # 4. Save statistical test results
        stats_summary = {
            "mann_whitney_stat": self.results.get('mann_whitney', {}).get('stat', None),
            "mann_whitney_pvalue": self.results.get('mann_whitney', {}).get('p_value', None),
            "pointbiserial_correlation": self.results.get('correlation', {}).get('r', None),
            "pointbiserial_pvalue": self.results.get('correlation', {}).get('p', None),
            "time_to_merge_correlation": self.results.get('time_to_merge_correlation', {}).get('r', None),
            "time_to_merge_pvalue": self.results.get('time_to_merge_correlation', {}).get('p', None),
            "iterations_correlation": self.results.get('iterations_correlation', {}).get('r', None),
            "iterations_pvalue": self.results.get('iterations_correlation', {}).get('p', None),
            "kruskal_wallis_stat": self.results.get('kruskal_wallis_categories', {}).get('stat', None),
            "kruskal_wallis_pvalue": self.results.get('kruskal_wallis_categories', {}).get('p_value', None),
            "chi2_category_agent": self.results.get('chi2_category_agent', {}).get('chi2', None),
            "chi2_category_agent_pvalue": self.results.get('chi2_category_agent', {}).get('p_value', None),
        }
        stats_df = pd.DataFrame([stats_summary])
        stats_path = os.path.join(data_dir, "statistical_tests.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"   ✓ Saved statistical tests: {stats_path}")

        # 5. Save topic model (pickle)
        if hasattr(self, 'topic_model'):
            model_path = os.path.join(models_dir, "bertopic_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.topic_model, f)
            print(f"   ✓ Saved BERTopic model: {model_path}")

        # 6. Save full results dictionary
        results_path = os.path.join(data_dir, "full_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"   ✓ Saved full results: {results_path}")

        # 7. Create summary report
        summary_path = os.path.join(self.run_dir, "SUMMARY.txt")
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FRICTION ANALYSIS - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Comments Analyzed: {len(self.analyzed_df)}\n\n")

            f.write("Agent Distribution:\n")
            for agent, count in self.analyzed_df['agent'].value_counts().items():
                f.write(f"  - {agent}: {count}\n")

            f.write("\nFriction Statistics (Mean Friction Score):\n")
            if 'friction_stats' in self.results:
                for _, row in self.results['friction_stats'].iterrows():
                    f.write(f"  - {row['agent']}: {row['mean']:.4f} (n={row['count']})\n")

            f.write("\nStatistical Tests:\n")
            if 'mann_whitney' in self.results:
                f.write(f"  Mann-Whitney U Test (Human vs AI):\n")
                f.write(f"    Statistic: {self.results['mann_whitney']['stat']:.2f}\n")
                f.write(f"    P-value: {self.results['mann_whitney']['p_value']:.4f}\n")
                f.write(f"    Significant: {'Yes' if self.results['mann_whitney']['p_value'] < 0.05 else 'No'}\n")

            if 'correlation' in self.results:
                f.write(f"  Point-Biserial Correlation (Friction vs Merge):\n")
                f.write(f"    Correlation: {self.results['correlation']['r']:.4f}\n")
                f.write(f"    P-value: {self.results['correlation']['p']:.4f}\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f"   ✓ Saved summary report: {summary_path}")
        print(f"\n✅ All results saved to: {self.run_dir}")

    def run_full_pipeline(self):
        """
        Esegue l'intera pipeline di analisi dall'inizio alla fine.
        Include tutte le Research Questions (RQ1-5) e analisi avanzate.
        """
        print("\n" + "=" * 70)
        print("FRICTION ANALYSIS PIPELINE - PRODUCTION MODE")
        print("=" * 70 + "\n")

        # Phase 0: Schema inspection
        self.inspect_dataset_schema()

        # Phase 1: Data loading and preprocessing
        self.load_data()
        self.preprocess_data()

        # Phase 2: Sentiment analysis
        self.analyze_sentiment()

        # Phase 3: Topic modeling
        self.extract_friction_topics()

        # Phase 3b: Category classification (Enhanced RQ2)
        self.classify_friction_categories()

        # Phase 4: Statistical analysis (RQ1, RQ3, RQ4)
        self.analyze_outcomes()

        # Phase 4b: Category-based friction analysis
        self.analyze_category_friction()

        # Enhanced Research Questions
        self.analyze_temporal_evolution()  # RQ5
        self.analyze_topic_agent_interaction()  # Enhanced RQ2
        self.analyze_timemerge_iterations()  # Enhanced RQ4

        # Phase 5 & 6: Visualization and saving
        self.visualize_results()
        self.visualize_categories()  # Category visualizations
        self.save_results()

        print("\n" + "=" * 70)
        print("FRICTION ANALYSIS PIPELINE - COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\n📊 Results available at: {self.run_dir}")
        print(f"   - Plots: {os.path.join(self.run_dir, 'plots')}")
        print(f"   - Data: {os.path.join(self.run_dir, 'data')}")
        print(f"   - Summary: {os.path.join(self.run_dir, 'SUMMARY.txt')}\n")

# Esecuzione
if __name__ == "__main__":
    project = FrictionAnalyzerProject()
    project.run_full_pipeline()
