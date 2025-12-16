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
        Inspect AIDev dataset to verify columns and structure
        before proceeding with main analysis.
        """
        print("=" * 70)
        print(">>> Phase 0: Dataset Schema Inspection")
        print("=" * 70)

        try:
            # Load a sample of each subset to inspect schema
            print("\n1. Inspecting pull_request (pre-filtered for 100+ stars repos)...")
            ds_pr = load_dataset("hao-li/AIDev", name="pull_request", split="train", streaming=True)
            sample_pr = next(iter(ds_pr))
            print(f"   Columns: {list(sample_pr.keys())}")
            print(f"   Sample keys with values:")
            for key in list(sample_pr.keys())[:10]:
                print(f"      {key}: {sample_pr[key]}")

            print("\n2. Inspecting pr_review_comments_v2...")
            ds_comments = load_dataset("hao-li/AIDev", name="pr_review_comments_v2", split="train", streaming=True)
            sample_comment = next(iter(ds_comments))
            print(f"   Columns: {list(sample_comment.keys())}")
            print(f"   Sample keys with values:")
            for key in list(sample_comment.keys())[:10]:
                print(f"      {key}: {sample_comment[key]}")

            print("\n" + "=" * 70)
            print("Schema inspection complete. Proceeding with data loading...")
            print("=" * 70 + "\n")

        except Exception as e:
            print(f"Warning during schema inspection: {e}")
            print("Continuing with pipeline...\n")

    # ==========================================
    # PHASE 1: Data Extraction & Preprocessing
    # ==========================================
    def load_data(self):
        """
        Downloads the AIDev dataset from Hugging Face and loads necessary tables.
        Uses the 'pull_request' subset which already contains only repos with 100+ GitHub stars.
        Also loads reviews and PR task types for comprehensive analysis.
        """
        print(">>> Phase 1: Loading AIDev dataset from Hugging Face...")

        try:
            # 1. Load AI Pull Requests metadata (pre-filtered for 100+ stars repos)
            print("Loading AI Pull Requests metadata (pre-filtered for 100+ stars repos)...")
            ds_pr = load_dataset("hao-li/AIDev", name="pull_request", split="train")
            self.data['prs'] = ds_pr.to_pandas()
            print(f"   Loaded {len(self.data['prs'])} AI PRs from 100+ stars repositories")

            # 2. Load Review Comments (inline comments on code)
            print("Loading Review Comments (inline)...")
            ds_comments = load_dataset("hao-li/AIDev", name="pr_review_comments_v2", split="train")
            self.data['comments'] = ds_comments.to_pandas()
            print(f"   Loaded {len(self.data['comments'])} review comments")

            # 3. Load Reviews (top-level review summaries)
            print("Loading Reviews (top-level)...")
            ds_reviews = load_dataset("hao-li/AIDev", name="pr_reviews", split="train")
            self.data['reviews'] = ds_reviews.to_pandas()
            print(f"   Loaded {len(self.data['reviews'])} reviews")

            # 4. Load PR Task Types (fix, feat, docs, etc.)
            print("Loading PR Task Types...")
            ds_task_types = load_dataset("hao-li/AIDev", name="pr_task_type", split="train")
            self.data['task_types'] = ds_task_types.to_pandas()
            print(f"   Loaded {len(self.data['task_types'])} PR task type classifications")

            print(f"\nData Loading Complete:")
            print(f"  - AI PRs (100+ stars repos): {len(self.data['prs'])}")
            print(f"  - Review Comments (inline): {len(self.data['comments'])}")
            print(f"  - Reviews (top-level): {len(self.data['reviews'])}")
            print(f"  - PR Task Types: {len(self.data['task_types'])}")

            # Validation: Check that dataframes are not empty
            if len(self.data['prs']) == 0 or len(self.data['comments']) == 0:
                raise ValueError("ERROR: Loaded empty dataframes! Check dataset access.")

        except Exception as e:
            print(f"Error loading data: {e}")
            print("Check internet connection and HuggingFace dataset access.")
            raise

    def preprocess_data(self):
        """
        Cleans text, filters bots, merges dataframes.
        - Filters PRs closed in less than 1 minute
        - Processes both inline comments and top-level reviews
        - Adds PR task type classification
        """
        print(">>> Phase 1b: Preprocessing & Filtering...")

        df_prs = self.data['prs'].copy()
        df_comments = self.data['comments'].copy()
        df_reviews = self.data['reviews'].copy()
        df_task_types = self.data['task_types'].copy()

        print(f"\nInspecting columns for join keys...")
        print(f"  AI PRs columns: {list(df_prs.columns)[:15]}...")
        print(f"  Comments columns: {list(df_comments.columns)[:15]}...")
        print(f"  Reviews columns: {list(df_reviews.columns)[:15]}...")

        # === STEP 1: Filter PRs closed in less than 1 minute ===
        print("\n1. Filtering PRs closed in less than 1 minute...")

        prs_before = len(df_prs)
        df_prs['created_at'] = pd.to_datetime(df_prs['created_at'], errors='coerce')
        df_prs['closed_at'] = pd.to_datetime(df_prs['closed_at'], errors='coerce')

        # Calculate duration in seconds
        df_prs['pr_duration_seconds'] = (df_prs['closed_at'] - df_prs['created_at']).dt.total_seconds()

        # Filter out PRs closed in less than 60 seconds (1 minute)
        df_prs = df_prs[(df_prs['pr_duration_seconds'].isna()) | (df_prs['pr_duration_seconds'] >= 60)]
        print(f"   ✓ Filtered {prs_before - len(df_prs)} PRs closed in < 1 minute")
        print(f"   Remaining PRs: {len(df_prs)}")

        # === STEP 2: Add PR Task Types ===
        print("\n2. Adding PR task types (fix, feat, docs, etc.)...")

        # Merge task types with PRs using PR id
        df_prs = pd.merge(
            df_prs,
            df_task_types[['id', 'type', 'confidence']],
            on='id',
            how='left'
        )
        df_prs['pr_type'] = df_prs['type'].fillna('unknown')

        type_counts = df_prs['pr_type'].value_counts()
        print(f"   PR type distribution:")
        for pr_type, count in type_counts.head(10).items():
            print(f"      {pr_type}: {count}")

        # === STEP 3: Extract repo info for merging ===
        print("\n3. Extracting repository information for merging...")

        # Extract PR identification from comments
        if 'pull_request_url' in df_comments.columns:
            def extract_pr_info(url):
                if pd.isna(url) or not isinstance(url, str):
                    return None, None
                parts = url.split('/')
                if len(parts) >= 7 and 'repos' in parts:
                    owner_repo = f"{parts[-4]}/{parts[-3]}"
                    pr_number = parts[-1]
                    return owner_repo, pr_number
                return None, None

            df_comments[['repo_path', 'pr_number']] = df_comments['pull_request_url'].apply(
                lambda x: pd.Series(extract_pr_info(x))
            )
            df_comments['pr_number'] = pd.to_numeric(df_comments['pr_number'], errors='coerce')

        # Extract repo info from PRs
        if 'repo_url' in df_prs.columns:
            def extract_repo_path(url):
                if pd.isna(url) or not isinstance(url, str):
                    return None
                parts = url.split('/')
                if len(parts) >= 5 and 'repos' in parts:
                    return f"{parts[-2]}/{parts[-1]}"
                return None

            df_prs['repo_path'] = df_prs['repo_url'].apply(extract_repo_path)

        # === STEP 4: Merge Comments with PRs ===
        print("\n4. Merging inline comments with PRs...")

        merged_comments = pd.merge(
            df_comments,
            df_prs,
            left_on=['repo_path', 'pr_number'],
            right_on=['repo_path', 'number'],
            how='inner',
            suffixes=('_comment', '_pr')
        )
        merged_comments['source'] = 'comment'  # Mark source type
        print(f"   ✓ Merged comments: {len(merged_comments)} comment-PR pairs")

        # === STEP 5: Merge Reviews with PRs ===
        print("\n5. Merging top-level reviews with PRs...")

        # Reviews use pr_id to reference the PR
        merged_reviews = pd.merge(
            df_reviews,
            df_prs,
            left_on='pr_id',
            right_on='id',
            how='inner',
            suffixes=('_review', '_pr')
        )
        merged_reviews['source'] = 'review'  # Mark source type
        print(f"   ✓ Merged reviews: {len(merged_reviews)} review-PR pairs")

        # === STEP 6: Filter Bots from Comments ===
        print("\n6. Filtering bot comments...")

        user_col = None
        for possible_name in ['user_name', 'user', 'author', 'login', 'user_comment', 'author_comment']:
            if possible_name in merged_comments.columns:
                user_col = possible_name
                break

        if user_col:
            bot_patterns = [r'\[bot\]', r'jenkins', r'ci/cd', r'linter', r'coverage', r'dependabot', r'coderabbit', r'copilot']
            merged_comments['is_bot'] = merged_comments[user_col].apply(
                lambda x: any(re.search(p, str(x).lower()) for p in bot_patterns) if pd.notnull(x) else False
            )
            before_count = len(merged_comments)
            merged_comments = merged_comments[~merged_comments['is_bot']]
            print(f"   ✓ Filtered {before_count - len(merged_comments)} bot comments")

        # === STEP 7: Filter Bots from Reviews ===
        print("\n7. Filtering bot reviews...")

        if 'user' in merged_reviews.columns:
            merged_reviews['is_bot'] = merged_reviews['user'].apply(
                lambda x: any(re.search(p, str(x).lower()) for p in bot_patterns) if pd.notnull(x) else False
            )
            # Also filter by user_type if available
            if 'user_type' in merged_reviews.columns:
                merged_reviews['is_bot'] = merged_reviews['is_bot'] | (merged_reviews['user_type'] == 'Bot')
            before_count = len(merged_reviews)
            merged_reviews = merged_reviews[~merged_reviews['is_bot']]
            print(f"   ✓ Filtered {before_count - len(merged_reviews)} bot reviews")

        # === STEP 8: Text Cleaning ===
        print("\n8. Cleaning text...")

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

        # Clean comments
        body_col_comments = None
        for possible_name in ['body', 'body_comment', 'comment', 'text', 'content']:
            if possible_name in merged_comments.columns:
                body_col_comments = possible_name
                break

        if body_col_comments:
            merged_comments['clean_body'] = merged_comments[body_col_comments].apply(clean_text)
            before_count = len(merged_comments)
            merged_comments = merged_comments[merged_comments['clean_body'].str.len() > 10]
            print(f"   ✓ Comments: Removed {before_count - len(merged_comments)} empty/too-short")

        # Clean reviews
        body_col_reviews = None
        for possible_name in ['body', 'body_review', 'review', 'text', 'content']:
            if possible_name in merged_reviews.columns:
                body_col_reviews = possible_name
                break

        if body_col_reviews:
            merged_reviews['clean_body'] = merged_reviews[body_col_reviews].apply(clean_text)
            before_count = len(merged_reviews)
            merged_reviews = merged_reviews[merged_reviews['clean_body'].str.len() > 10]
            print(f"   ✓ Reviews: Removed {before_count - len(merged_reviews)} empty/too-short")

        # === STEP 9: Standardize columns and combine ===
        print("\n9. Standardizing and combining datasets...")

        # Ensure agent column exists
        for df in [merged_comments, merged_reviews]:
            if 'agent' not in df.columns:
                df['agent'] = 'Unknown_AI'
            else:
                df['agent'] = df['agent'].fillna('Unknown_AI')

            if 'pr_type' not in df.columns:
                df['pr_type'] = 'unknown'

        # Select common columns for combined dataset
        common_cols = ['clean_body', 'agent', 'source', 'pr_type', 'created_at', 'closed_at', 'state']

        # Add id columns for tracking
        if 'id_pr' in merged_comments.columns:
            common_cols.append('id_pr')
        elif 'id' in merged_comments.columns:
            merged_comments['id_pr'] = merged_comments['id']
            common_cols.append('id_pr')

        if 'id_pr' not in merged_reviews.columns and 'id_pr' in merged_reviews.columns:
            pass
        elif 'pr_id' in merged_reviews.columns:
            merged_reviews['id_pr'] = merged_reviews['pr_id']

        # Filter to available columns
        available_cols_comments = [c for c in common_cols if c in merged_comments.columns]
        available_cols_reviews = [c for c in common_cols if c in merged_reviews.columns]

        # Store separate datasets
        self.comments_df = merged_comments.copy()
        self.reviews_df = merged_reviews.copy()

        # Create combined dataset
        combined = pd.concat([
            merged_comments[available_cols_comments],
            merged_reviews[available_cols_reviews]
        ], ignore_index=True)

        # === STEP 10: Validation ===
        print("\n10. Validating datasets...")

        print(f"   ✓ Comments dataset: {len(self.comments_df)} rows")
        print(f"   ✓ Reviews dataset: {len(self.reviews_df)} rows")
        print(f"   ✓ Combined dataset: {len(combined)} rows")

        if len(combined) == 0:
            raise ValueError("❌ ERROR: No data remained after preprocessing!")

        # Check agent distribution
        agent_counts = combined['agent'].value_counts()
        print(f"\n   Agent distribution (combined):")
        for agent, count in agent_counts.items():
            print(f"      {agent}: {count}")

        # Check source distribution
        source_counts = combined['source'].value_counts()
        print(f"\n   Source distribution:")
        for source, count in source_counts.items():
            print(f"      {source}: {count}")

        # Check PR type distribution
        type_counts = combined['pr_type'].value_counts()
        print(f"\n   PR type distribution:")
        for pr_type, count in type_counts.head(10).items():
            print(f"      {pr_type}: {count}")

        self.dataset = combined
        print(f"\n✅ Preprocessing Complete.")
        print(f"   - Comments: {len(self.comments_df)}")
        print(f"   - Reviews: {len(self.reviews_df)}")
        print(f"   - Combined: {len(self.dataset)}")
        print(f"   Ready for sentiment analysis!")

    # ==========================================
    # PHASE 2: Sentiment Analysis (Friction)
    # ==========================================
    def analyze_sentiment(self):
        """
        Applica RoBERTa pre-trained per calcolare sentiment.
        Analyzes both combined dataset and separate comments/reviews.
        Friction Score = Probabilità Negativa.
        """
        print(">>> Phase 2: Running Sentiment Analysis (RoBERTa)...")

        device = 0 if torch.cuda.is_available() else -1
        print(f"   Using device: {'GPU' if device == 0 else 'CPU'}")
        sentiment_pipe = pipeline("sentiment-analysis", model=self.models['sentiment'], tokenizer=self.models['sentiment'], device=device, top_k=None)

        batch_size = 32

        def run_sentiment_analysis(df, name):
            """Helper to run sentiment analysis on a dataframe"""
            texts = df['clean_body'].tolist()
            print(f"\n   Processing {len(texts)} {name} in batches of {batch_size}...")

            results = []
            for i in tqdm(range(0, len(texts), batch_size), desc=f"   {name}"):
                batch = texts[i:i+batch_size]
                preds = sentiment_pipe(batch, truncation=True, max_length=512)
                results.extend(preds)

            friction_scores = []
            sentiments = []

            for res in results:
                neg_score = next((item['score'] for item in res if item['label'] == 'negative'), 0)
                label = max(res, key=lambda x: x['score'])['label']
                friction_scores.append(neg_score)
                sentiments.append(label)

            df_copy = df.iloc[:len(texts)].copy()
            df_copy['friction_score'] = friction_scores
            df_copy['sentiment_label'] = sentiments
            df_copy['is_negative'] = df_copy['sentiment_label'] == 'negative'

            return df_copy

        # Analyze combined dataset
        print("\n   === Analyzing Combined Dataset ===")
        self.analyzed_df = run_sentiment_analysis(self.dataset, "combined items")

        # Analyze comments separately
        print("\n   === Analyzing Comments ===")
        self.analyzed_comments_df = run_sentiment_analysis(self.comments_df, "comments")

        # Analyze reviews separately
        print("\n   === Analyzing Reviews ===")
        self.analyzed_reviews_df = run_sentiment_analysis(self.reviews_df, "reviews")

        # Summary statistics
        print("\n   === Sentiment Analysis Summary ===")
        print(f"   Combined: {len(self.analyzed_df)} items")
        print(f"      - Negative: {self.analyzed_df['is_negative'].sum()} ({100*self.analyzed_df['is_negative'].mean():.1f}%)")
        print(f"      - Mean friction: {self.analyzed_df['friction_score'].mean():.3f}")

        print(f"\n   Comments: {len(self.analyzed_comments_df)} items")
        print(f"      - Negative: {self.analyzed_comments_df['is_negative'].sum()} ({100*self.analyzed_comments_df['is_negative'].mean():.1f}%)")
        print(f"      - Mean friction: {self.analyzed_comments_df['friction_score'].mean():.3f}")

        print(f"\n   Reviews: {len(self.analyzed_reviews_df)} items")
        print(f"      - Negative: {self.analyzed_reviews_df['is_negative'].sum()} ({100*self.analyzed_reviews_df['is_negative'].mean():.1f}%)")
        print(f"      - Mean friction: {self.analyzed_reviews_df['friction_score'].mean():.3f}")

        print("\nSentiment Analysis Complete.")

    # ==========================================
    # PHASE 3: Topic Modeling (BERTopic)
    # ==========================================
    def extract_friction_topics(self):
        """
        Apply BERTopic on negative comments to identify friction causes.
        RQ3: Which specific topics generate the most friction?

        Uses best practices:
        - CountVectorizer with stopwords removal AFTER clustering
        - ClassTfidfTransformer to reduce frequent word impact
        - N-gram range (1,2) for better topic representations
        """
        print(">>> Phase 3: Topic Modeling on Negative Comments...")

        from sklearn.feature_extraction.text import CountVectorizer
        from bertopic.vectorizers import ClassTfidfTransformer

        negative_comments = self.analyzed_df[self.analyzed_df['is_negative']]['clean_body'].tolist()

        if len(negative_comments) < 10:
            print("Not enough negative comments for Topic Modeling.")
            return

        print(f"   Processing {len(negative_comments)} negative comments...")

        # CountVectorizer to remove stopwords AFTER clustering (BERTopic best practice)
        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=2,
            ngram_range=(1, 2)
        )

        # ClassTfidfTransformer to reduce impact of frequent words
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        topic_model = BERTopic(
            embedding_model=self.models['topic_embedding'],
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            min_topic_size=5,
            nr_topics="auto"
        )

        topics, probs = topic_model.fit_transform(negative_comments)

        # Extract topic info
        topic_info = topic_model.get_topic_info()
        print("Top Friction Topics identified:")
        print(topic_info.head(10))

        # Save topic map for future use
        self.results['topics'] = topic_info
        self.topic_model = topic_model

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
        RQ1, RQ2, RQ4: Aggregate metrics and calculate statistics.
        Focus on comparing AI agents (no Human baseline).
        """
        print(">>> Phase 4: Statistical Analysis...")
        df = self.analyzed_df

        # RQ1: Friction by AI Agent
        friction_by_agent = df.groupby('agent')['friction_score'].agg(['mean', 'count', 'std']).reset_index()
        self.results['friction_stats'] = friction_by_agent

        print("\n   Friction Statistics by Agent:")
        print(friction_by_agent.to_string(index=False))

        # RQ2: Statistical Test - Kruskal-Wallis across AI agents
        print("\n   Kruskal-Wallis test across AI agents...")
        agent_groups = [group['friction_score'].values for name, group in df.groupby('agent')]

        if len(agent_groups) >= 2 and all(len(g) >= 2 for g in agent_groups):
            try:
                stat, p_val = stats.kruskal(*agent_groups)
                self.results['kruskal_wallis_agents'] = {'stat': stat, 'p_value': p_val}
                sig = "Yes" if p_val < 0.05 else "No"
                print(f"   Kruskal-Wallis test: H={stat:.2f}, p={p_val:.4f}, Significant: {sig}")

                # Post-hoc pairwise comparisons if significant
                if p_val < 0.05:
                    print("\n   Pairwise Mann-Whitney U tests (post-hoc):")
                    agents = df['agent'].unique()
                    pairwise_results = []
                    for i in range(len(agents)):
                        for j in range(i + 1, len(agents)):
                            a1, a2 = agents[i], agents[j]
                            g1 = df[df['agent'] == a1]['friction_score']
                            g2 = df[df['agent'] == a2]['friction_score']
                            if len(g1) >= 2 and len(g2) >= 2:
                                mw_stat, mw_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                                pairwise_results.append({
                                    'pair': f"{a1} vs {a2}",
                                    'stat': mw_stat,
                                    'p_value': mw_p
                                })
                                sig_mark = "*" if mw_p < 0.05 else ""
                                print(f"      {a1} vs {a2}: U={mw_stat:.0f}, p={mw_p:.4f} {sig_mark}")
                    self.results['pairwise_tests'] = pairwise_results
            except Exception as e:
                print(f"   Kruskal-Wallis test failed: {e}")
        else:
            print("   Insufficient data for Kruskal-Wallis test")
        
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
            print(f"\n   RQ4: Correlation Friction <-> Merge Success: r = {corr:.3f}, p = {p_val_corr:.4f}")

            # Merge rate by agent
            merge_rates = df.groupby('agent')['is_merged'].agg(['mean', 'count']).reset_index()
            merge_rates.columns = ['agent', 'merge_rate', 'count']
            self.results['merge_rates'] = merge_rates
            print("\n   Merge rates by agent:")
            for _, row in merge_rates.iterrows():
                print(f"      {row['agent']}: {row['merge_rate']*100:.1f}% (n={row['count']})")
        else:
            print("   Warning: No state/outcome column found for correlation analysis")

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
        Generate and save required plots.
        Focus on AI agents comparison (RQ1, RQ2).
        """
        print(">>> Phase 5: Visualization...")
        df = self.analyzed_df
        plots_dir = os.path.join(self.run_dir, "plots")

        # 1. Boxplot Friction by AI Agent
        plt.figure(figsize=(12, 7))
        order = df.groupby('agent')['friction_score'].mean().sort_values(ascending=False).index
        sns.boxplot(x='agent', y='friction_score', data=df, order=order, palette="viridis")
        plt.title("Friction Score by AI Agent (RQ1, RQ2)", fontsize=14, fontweight='bold')
        plt.ylabel("Friction Score (P(negative))", fontsize=12)
        plt.xlabel("AI Agent", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "friction_boxplot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 2. Sentiment Distribution by AI Agent
        plt.figure(figsize=(12, 7))
        sns.countplot(x='sentiment_label', hue='agent', data=df, palette="Set2",
                     order=['negative', 'neutral', 'positive'])
        plt.title("Sentiment Distribution Across AI Agents", fontsize=14, fontweight='bold')
        plt.xlabel("Sentiment", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.legend(title="AI Agent", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "sentiment_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 3. Friction Score Distribution (Histogram)
        plt.figure(figsize=(12, 7))
        for agent in df['agent'].unique():
            agent_data = df[df['agent'] == agent]['friction_score']
            plt.hist(agent_data, alpha=0.5, label=agent, bins=30)
        plt.xlabel("Friction Score", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Distribution of Friction Scores by AI Agent", fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "friction_distribution_histogram.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 4. Violin plot for more detailed distribution view
        plt.figure(figsize=(12, 7))
        sns.violinplot(x='agent', y='friction_score', data=df, order=order, palette="muted")
        plt.title("Friction Score Distribution by AI Agent (Violin Plot)", fontsize=14, fontweight='bold')
        plt.ylabel("Friction Score", fontsize=12)
        plt.xlabel("AI Agent", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "friction_violin.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
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
    # PHASE 5c: Source-based Visualizations (Comments vs Reviews)
    # ==========================================
    def visualize_by_source(self):
        """
        Generate visualizations comparing comments vs reviews.
        """
        print(">>> Phase 5c: Source-based Visualizations (Comments vs Reviews)...")
        plots_dir = os.path.join(self.run_dir, "plots")

        # Create subfolders
        comments_dir = os.path.join(plots_dir, "comments_only")
        reviews_dir = os.path.join(plots_dir, "reviews_only")
        os.makedirs(comments_dir, exist_ok=True)
        os.makedirs(reviews_dir, exist_ok=True)

        df = self.analyzed_df

        # === AGGREGATED: Comments vs Reviews comparison ===
        print("\n   Creating aggregated visualizations...")

        # 1. Friction by Source Type
        if 'source' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='source', y='friction_score', data=df, palette="Set2")
            plt.title("Friction Score: Comments vs Reviews", fontsize=14, fontweight='bold')
            plt.ylabel("Friction Score", fontsize=12)
            plt.xlabel("Source Type", fontsize=12)
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, "friction_by_source.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   Saved: {plot_path}")
            plt.close()

            # 2. Sentiment Distribution by Source
            plt.figure(figsize=(10, 6))
            sns.countplot(x='sentiment_label', hue='source', data=df, palette="Set2",
                         order=['negative', 'neutral', 'positive'])
            plt.title("Sentiment Distribution: Comments vs Reviews", fontsize=14, fontweight='bold')
            plt.xlabel("Sentiment", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.legend(title="Source")
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, "sentiment_by_source.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   Saved: {plot_path}")
            plt.close()

            # 3. Agent comparison by Source (grouped bar)
            plt.figure(figsize=(14, 7))
            source_agent = df.groupby(['agent', 'source'])['friction_score'].mean().unstack()
            source_agent.plot(kind='bar', figsize=(14, 7), colormap='Set2')
            plt.title("Mean Friction Score by Agent and Source", fontsize=14, fontweight='bold')
            plt.xlabel("AI Agent", fontsize=12)
            plt.ylabel("Mean Friction Score", fontsize=12)
            plt.legend(title="Source")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, "friction_agent_by_source.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   Saved: {plot_path}")
            plt.close()

        # === COMMENTS ONLY visualizations ===
        print("\n   Creating comments-only visualizations...")
        if hasattr(self, 'analyzed_comments_df') and len(self.analyzed_comments_df) > 0:
            df_comments = self.analyzed_comments_df

            # Friction by Agent (Comments only)
            plt.figure(figsize=(12, 7))
            if 'agent' in df_comments.columns:
                order = df_comments.groupby('agent')['friction_score'].mean().sort_values(ascending=False).index
                sns.boxplot(x='agent', y='friction_score', data=df_comments, order=order, palette="Blues")
                plt.title("Friction Score by AI Agent (Comments Only)", fontsize=14, fontweight='bold')
                plt.ylabel("Friction Score", fontsize=12)
                plt.xlabel("AI Agent", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_path = os.path.join(comments_dir, "friction_by_agent_comments.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"   Saved: {plot_path}")
                plt.close()

            # Sentiment Distribution (Comments only)
            plt.figure(figsize=(10, 6))
            sns.countplot(x='sentiment_label', data=df_comments, palette="Blues",
                         order=['negative', 'neutral', 'positive'])
            plt.title("Sentiment Distribution (Comments Only)", fontsize=14, fontweight='bold')
            plt.xlabel("Sentiment", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.tight_layout()
            plot_path = os.path.join(comments_dir, "sentiment_distribution_comments.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   Saved: {plot_path}")
            plt.close()

        # === REVIEWS ONLY visualizations ===
        print("\n   Creating reviews-only visualizations...")
        if hasattr(self, 'analyzed_reviews_df') and len(self.analyzed_reviews_df) > 0:
            df_reviews = self.analyzed_reviews_df

            # Friction by Agent (Reviews only)
            plt.figure(figsize=(12, 7))
            if 'agent' in df_reviews.columns:
                order = df_reviews.groupby('agent')['friction_score'].mean().sort_values(ascending=False).index
                sns.boxplot(x='agent', y='friction_score', data=df_reviews, order=order, palette="Oranges")
                plt.title("Friction Score by AI Agent (Reviews Only)", fontsize=14, fontweight='bold')
                plt.ylabel("Friction Score", fontsize=12)
                plt.xlabel("AI Agent", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_path = os.path.join(reviews_dir, "friction_by_agent_reviews.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"   Saved: {plot_path}")
                plt.close()

            # Sentiment Distribution (Reviews only)
            plt.figure(figsize=(10, 6))
            sns.countplot(x='sentiment_label', data=df_reviews, palette="Oranges",
                         order=['negative', 'neutral', 'positive'])
            plt.title("Sentiment Distribution (Reviews Only)", fontsize=14, fontweight='bold')
            plt.xlabel("Sentiment", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.tight_layout()
            plot_path = os.path.join(reviews_dir, "sentiment_distribution_reviews.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   Saved: {plot_path}")
            plt.close()

        print(f"   Source-based visualizations complete.")

    # ==========================================
    # PHASE 5d: PR Type Visualizations
    # ==========================================
    def visualize_by_pr_type(self):
        """
        Generate visualizations by PR type (fix, feat, docs, etc.)
        """
        print(">>> Phase 5d: PR Type Visualizations...")
        plots_dir = os.path.join(self.run_dir, "plots")
        pr_type_dir = os.path.join(plots_dir, "by_pr_type")
        os.makedirs(pr_type_dir, exist_ok=True)

        df = self.analyzed_df

        if 'pr_type' not in df.columns:
            print("   No PR type column found. Skipping PR type visualizations.")
            return

        # Filter out unknown types for cleaner visualizations
        df_typed = df[df['pr_type'] != 'unknown'].copy()

        if len(df_typed) == 0:
            print("   No typed PRs found. Skipping PR type visualizations.")
            return

        print(f"   Analyzing {len(df_typed)} items with known PR types...")

        # 1. Friction by PR Type (Boxplot)
        plt.figure(figsize=(14, 7))
        order = df_typed.groupby('pr_type')['friction_score'].mean().sort_values(ascending=False).index
        sns.boxplot(x='pr_type', y='friction_score', data=df_typed, order=order, palette="husl")
        plt.title("Friction Score by PR Type", fontsize=14, fontweight='bold')
        plt.ylabel("Friction Score", fontsize=12)
        plt.xlabel("PR Type", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(pr_type_dir, "friction_by_pr_type.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 2. Sentiment Distribution by PR Type
        plt.figure(figsize=(14, 7))
        sns.countplot(x='pr_type', hue='sentiment_label', data=df_typed,
                     hue_order=['negative', 'neutral', 'positive'],
                     order=order, palette="Set1")
        plt.title("Sentiment Distribution by PR Type", fontsize=14, fontweight='bold')
        plt.xlabel("PR Type", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.legend(title="Sentiment")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(pr_type_dir, "sentiment_by_pr_type.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 3. PR Type Distribution (Pie Chart)
        plt.figure(figsize=(10, 8))
        type_counts = df_typed['pr_type'].value_counts()
        colors = plt.cm.tab20(np.linspace(0, 1, len(type_counts)))
        plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title("Distribution of PR Types", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(pr_type_dir, "pr_type_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 4. Heatmap: PR Type vs Agent
        plt.figure(figsize=(14, 8))
        type_agent = df_typed.groupby(['pr_type', 'agent'])['friction_score'].mean().unstack()
        sns.heatmap(type_agent, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Mean Friction'})
        plt.title("Mean Friction Score: PR Type vs AI Agent", fontsize=14, fontweight='bold')
        plt.xlabel("AI Agent", fontsize=12)
        plt.ylabel("PR Type", fontsize=12)
        plt.tight_layout()
        plot_path = os.path.join(pr_type_dir, "friction_heatmap_type_agent.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 5. Negative Sentiment Rate by PR Type
        plt.figure(figsize=(12, 6))
        neg_rate = df_typed.groupby('pr_type')['is_negative'].mean().sort_values(ascending=False)
        neg_rate.plot(kind='bar', color='coral', edgecolor='black')
        plt.title("Negative Sentiment Rate by PR Type", fontsize=14, fontweight='bold')
        plt.xlabel("PR Type", fontsize=12)
        plt.ylabel("Proportion of Negative Sentiment", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.axhline(y=df_typed['is_negative'].mean(), color='red', linestyle='--', label='Overall Mean')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(pr_type_dir, "negative_rate_by_pr_type.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 6. PR Type by Source (Comments vs Reviews)
        if 'source' in df_typed.columns:
            plt.figure(figsize=(14, 7))
            type_source = df_typed.groupby(['pr_type', 'source'])['friction_score'].mean().unstack()
            type_source.plot(kind='bar', figsize=(14, 7), colormap='Set2')
            plt.title("Mean Friction by PR Type and Source", fontsize=14, fontweight='bold')
            plt.xlabel("PR Type", fontsize=12)
            plt.ylabel("Mean Friction Score", fontsize=12)
            plt.legend(title="Source")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_path = os.path.join(pr_type_dir, "friction_type_by_source.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   Saved: {plot_path}")
            plt.close()

        # Store PR type statistics
        pr_type_stats = df_typed.groupby('pr_type').agg({
            'friction_score': ['mean', 'std', 'count'],
            'is_negative': 'mean'
        }).round(4)
        pr_type_stats.columns = ['mean_friction', 'std_friction', 'count', 'negative_rate']
        self.results['pr_type_stats'] = pr_type_stats.reset_index()

        print(f"   PR type visualizations complete.")

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

        # 1. Save analyzed combined dataset
        analyzed_path = os.path.join(data_dir, "analyzed_combined.csv")
        self.analyzed_df.to_csv(analyzed_path, index=False)
        print(f"   ✓ Saved analyzed combined: {analyzed_path}")

        # 1b. Save analyzed comments separately
        if hasattr(self, 'analyzed_comments_df'):
            comments_path = os.path.join(data_dir, "analyzed_comments_only.csv")
            self.analyzed_comments_df.to_csv(comments_path, index=False)
            print(f"   ✓ Saved analyzed comments: {comments_path}")

        # 1c. Save analyzed reviews separately
        if hasattr(self, 'analyzed_reviews_df'):
            reviews_path = os.path.join(data_dir, "analyzed_reviews_only.csv")
            self.analyzed_reviews_df.to_csv(reviews_path, index=False)
            print(f"   ✓ Saved analyzed reviews: {reviews_path}")

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
            "kruskal_wallis_agents_stat": self.results.get('kruskal_wallis_agents', {}).get('stat', None),
            "kruskal_wallis_agents_pvalue": self.results.get('kruskal_wallis_agents', {}).get('p_value', None),
            "pointbiserial_correlation": self.results.get('correlation', {}).get('r', None),
            "pointbiserial_pvalue": self.results.get('correlation', {}).get('p', None),
            "time_to_merge_correlation": self.results.get('time_to_merge_correlation', {}).get('r', None),
            "time_to_merge_pvalue": self.results.get('time_to_merge_correlation', {}).get('p', None),
            "iterations_correlation": self.results.get('iterations_correlation', {}).get('r', None),
            "iterations_pvalue": self.results.get('iterations_correlation', {}).get('p', None),
            "kruskal_wallis_categories_stat": self.results.get('kruskal_wallis_categories', {}).get('stat', None),
            "kruskal_wallis_categories_pvalue": self.results.get('kruskal_wallis_categories', {}).get('p_value', None),
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
            f.write("AI Agents Code Review Friction Analysis\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Comments Analyzed: {len(self.analyzed_df)}\n")
            f.write(f"Filter: Repositories with 100+ GitHub stars\n\n")

            f.write("AI Agent Distribution:\n")
            for agent, count in self.analyzed_df['agent'].value_counts().items():
                f.write(f"  - {agent}: {count}\n")

            f.write("\nFriction Statistics (Mean Friction Score):\n")
            if 'friction_stats' in self.results:
                for _, row in self.results['friction_stats'].iterrows():
                    f.write(f"  - {row['agent']}: {row['mean']:.4f} (n={int(row['count'])})\n")

            f.write("\nStatistical Tests:\n")
            if 'kruskal_wallis_agents' in self.results:
                f.write(f"  Kruskal-Wallis Test (AI Agents Comparison):\n")
                f.write(f"    H-statistic: {self.results['kruskal_wallis_agents']['stat']:.2f}\n")
                f.write(f"    P-value: {self.results['kruskal_wallis_agents']['p_value']:.4f}\n")
                sig = 'Yes' if self.results['kruskal_wallis_agents']['p_value'] < 0.05 else 'No'
                f.write(f"    Significant: {sig}\n")

            if 'correlation' in self.results:
                f.write(f"\n  Point-Biserial Correlation (Friction vs Merge):\n")
                f.write(f"    Correlation: {self.results['correlation']['r']:.4f}\n")
                f.write(f"    P-value: {self.results['correlation']['p']:.4f}\n")

            if 'kruskal_wallis_categories' in self.results:
                f.write(f"\n  Kruskal-Wallis Test (Friction Categories):\n")
                f.write(f"    H-statistic: {self.results['kruskal_wallis_categories']['stat']:.2f}\n")
                f.write(f"    P-value: {self.results['kruskal_wallis_categories']['p_value']:.4f}\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f"   Saved summary report: {summary_path}")
        print(f"\nAll results saved to: {self.run_dir}")

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
        self.visualize_by_source()  # Comments vs Reviews visualizations
        self.visualize_by_pr_type()  # PR Type visualizations
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
