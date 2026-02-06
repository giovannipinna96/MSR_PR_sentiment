import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import pipeline
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from scipy import stats
import re
from tqdm import tqdm
tqdm.pandas()  # Enable progress_apply for pandas
import warnings
import os
from datetime import datetime
import pickle
import torch
from fast_langdetect import detect_language  # Language detection for filtering non-English

# Statistical testing imports
import scikit_posthocs as sp  # Dunn's test for proper post-hoc after Kruskal-Wallis
from statsmodels.stats.multitest import multipletests  # Multiple comparison corrections
from statsmodels.regression.linear_model import OLS  # For confounding variable control
from statsmodels.tools import add_constant  # For regression intercept
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower  # Power analysis
from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa  # Inter-rater agreement
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For encoding categorical variables
from sklearn.metrics import cohen_kappa_score  # Simpler Cohen's Kappa implementation

# SentiCR - Code Review specific sentiment analysis
from senticr import SentiCR, create_senticr_pipeline

# Configurazione
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

class FrictionAnalyzerProject:
    def __init__(self):
        # GPU/CPU device detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_id = 0 if torch.cuda.is_available() else -1
        print(f"🖥️  Device: {self.device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else ""))

        # Primary sentiment model
        self.models = {
            "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "topic_embedding": "all-MiniLM-L6-v2",
            "category_classifier": "facebook/bart-large-mnli"  # Zero-shot classification
        }

        # Pre-load SentenceTransformer on GPU for BERTopic
        print(f"   Loading SentenceTransformer ({self.models['topic_embedding']}) on {self.device.upper()}...")
        self.embedding_model = SentenceTransformer(self.models['topic_embedding'], device=self.device)

        # Multi-model sentiment analysis configuration
        # ============================================================
        # RATIONALE: Using multiple models increases robustness by:
        # 1. Reducing model-specific biases
        # 2. Enabling inter-model agreement analysis (Cohen's Kappa)
        # 3. Providing confidence through consensus
        #
        # IMPORTANT: All models must have the SAME number of classes (3)
        # for valid inter-model agreement calculation. Binary models
        # (2 classes) cannot be compared with ternary models (3 classes)
        # because Cohen's Kappa requires identical label sets.
        #
        # Models selected (all 3-class: negative/neutral/positive):
        # - Twitter RoBERTa: Best for short, informal text (tweets ≈ comments)
        # - BERTweet Sentiment: Trained on tweets, 3-class output
        # - CardiffNLP RoBERTa: Another 3-class Twitter sentiment model
        #
        # References:
        # - arXiv:2401.10845 - Emotion Classification in SE (2024)
        # - PMC11132486 - Fuzzy Ensemble BERT for SE (2024)
        # ============================================================
        # Primary sentiment models by language
        # Each major language uses a specialized model, others use multilingual fallback
        # All models are 3-class (negative/neutral/positive) for consistent comparison
        self.sentiment_models_by_lang = {
            "EN": {
                "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "description": "RoBERTa trained on 124M tweets, fine-tuned on TweetEval (3-class, English)",
                "labels": {"negative": "negative", "neutral": "neutral", "positive": "positive"},
            },
            "FR": {
                "model_id": "cmarkea/distilcamembert-base-sentiment",
                "description": "DistilCamemBERT fine-tuned for French sentiment (5-star → 3-class)",
                "map_to_3class": True,  # Enable 5-class to 3-class mapping
                "labels": {
                    # Model outputs star ratings - map to 3-class
                    "1 star": "negative", "2 stars": "negative",
                    "3 stars": "neutral",
                    "4 stars": "positive", "5 stars": "positive",
                    # Fallback mappings if model outputs different format
                    "positive": "positive", "negative": "negative", "neutral": "neutral",
                },
            },
            "DE": {
                "model_id": "oliverguhr/german-sentiment-bert",
                "description": "German BERT trained on 1.8M samples (3-class)",
                "labels": {"positive": "positive", "negative": "negative", "neutral": "neutral"},
            },
            "ES": {
                "model_id": "finiteautomata/beto-sentiment-analysis",
                "description": "BETO (Spanish BERT) fine-tuned on TASS 2020 (3-class: POS/NEG/NEU)",
                "labels": {"POS": "positive", "NEG": "negative", "NEU": "neutral"},
            },
            "IT": {
                "model_id": "neuraly/bert-base-italian-cased-sentiment",
                "description": "Italian BERT fine-tuned for sentiment (3-class)",
                "labels": {"positive": "positive", "negative": "negative", "neutral": "neutral"},
            },
            "PT": {
                "model_id": "lucas-leme/FinBERT-PT-BR",
                "description": "FinBERT fine-tuned for Brazilian Portuguese (3-class)",
                "labels": {"POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral"},
            },
            "ZH": {
                "model_id": "yiyanghkust/finbert-tone-chinese",
                "description": "FinBERT Chinese fine-tuned on financial news (3-class)",
                "labels": {"0": "neutral", "1": "positive", "2": "negative",
                          "Neutral": "neutral", "Positive": "positive", "Negative": "negative",
                          "neutral": "neutral", "positive": "positive", "negative": "negative"},
            },
            "JA": {
                "model_id": "koheiduck/bert-japanese-finetuned-sentiment",
                "description": "Japanese BERT fine-tuned for sentiment analysis",
                "labels": {"positive": "positive", "negative": "negative", "neutral": "neutral",
                          "POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral"},
            },
            "KO": {
                "model_id": "WhitePeak/bert-base-cased-Korean-sentiment",
                "description": "Korean BERT fine-tuned for sentiment (accuracy 0.92)",
                "labels": {"positive": "positive", "negative": "negative", "neutral": "neutral",
                          "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"},
            },
            "RU": {
                "model_id": "blanchefort/rubert-base-cased-sentiment",
                "description": "RuBERT for Russian sentiment (3-class)",
                "labels": {"NEUTRAL": "neutral", "POSITIVE": "positive", "NEGATIVE": "negative"},
            },
            "MULTILINGUAL": {
                "model_id": "clapAI/roberta-base-multilingual-sentiment",
                "description": "RoBERTa multilingual sentiment 2025 (16+ languages, 3-class)",
                "labels": {"positive": "positive", "negative": "negative", "neutral": "neutral"},
            }
        }

        # Embedding models for BERTopic by language
        # Each major language uses a specialized sentence embedding model
        # Fallback uses paraphrase-multilingual-MiniLM-L12-v2 (50+ languages)
        self.embedding_models_by_lang = {
            "EN": {
                "model_id": "all-MiniLM-L6-v2",
                "description": "English MiniLM (384 dim), fast and accurate for English",
            },
            "ZH": {
                "model_id": "shibing624/text2vec-base-chinese",
                "description": "Chinese MacBERT fine-tuned with CoSENT (768 dim)",
            },
            "JA": {
                "model_id": "pkshatech/GLuCoSE-base-ja",
                "description": "Japanese GLuCoSE sentence encoder (768 dim)",
            },
            "KO": {
                "model_id": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
                "description": "Korean SBERT trained on KLUE NLI + augmented STS (768 dim)",
            },
            "FR": {
                "model_id": "dangvantuan/sentence-camembert-large",
                "description": "French CamemBERT fine-tuned for sentence similarity (1024 dim)",
            },
            "DE": {
                "model_id": "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
                "description": "German/English cross-lingual RoBERTa (768 dim)",
            },
            "ES": {
                "model_id": "hiiamsid/sentence_similarity_spanish_es",
                "description": "Spanish BETO fine-tuned for sentence similarity (768 dim)",
            },
            "IT": {
                "model_id": "nickprock/sentence-bert-base-italian-uncased",
                "description": "Italian BERT fine-tuned for sentence similarity (768 dim)",
            },
            "PT": {
                "model_id": "rufimelo/bert-large-portuguese-cased-sts",
                "description": "Portuguese BERTimbau large for STS (1024 dim)",
            },
            "RU": {
                "model_id": "paraphrase-multilingual-MiniLM-L12-v2",
                "description": "Multilingual fallback for Russian (384 dim)",
            },
            "MULTILINGUAL": {
                "model_id": "paraphrase-multilingual-MiniLM-L12-v2",
                "description": "Multilingual MiniLM for 50+ languages (384 dim)",
            }
        }

        # Emotion models by language
        # English uses specialized model, others use multilingual XLM-EMO-T
        # ES removed: pysentimiento causes CUDA "index out of bounds" errors
        self.emotion_models_by_lang = {
            "EN": {
                "model_id": "j-hartmann/emotion-english-distilroberta-base",
                "description": "DistilRoBERTa Ekman 7 emotions (English)",
                "labels": ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
            },
            "MULTILINGUAL": {
                "model_id": "MilaNLProc/xlm-emo-t",
                "description": "XLM-EMO-T multilingual emotions (19 languages, 8 emotions)",
                "labels": ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"],
            }
        }

        # Validation models for multi-model robustness check (all 3-class)
        self.sentiment_models = {
            "twitter_roberta": {
                "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "description": "RoBERTa trained on 124M tweets, fine-tuned on TweetEval (3-class)",
                "labels": {"negative": "negative", "neutral": "neutral", "positive": "positive"},
                "primary": True  # Used as primary model for friction score
            },
            # bertweet_sentiment REMOVED: tokenizer incompatible with code review text
            # (causes "index out of range" errors due to code snippets, paths, special chars)
            "cardiffnlp_roberta": {
                "model_id": "cardiffnlp/twitter-roberta-base-sentiment",
                "description": "RoBERTa base trained on tweets for sentiment (3-class)",
                "labels": {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"},
                "primary": False
            }
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

            # 5. Load Human PRs as baseline for comparison
            print("Loading Human PRs baseline...")
            ds_human = load_dataset("hao-li/AIDev", name="human_pull_request", split="train")
            self.data['human_prs'] = ds_human.to_pandas()
            self.data['human_prs']['agent'] = 'Human'  # Label for comparison
            print(f"   Loaded {len(self.data['human_prs'])} human PRs as baseline")

            # 6. Load Human PR Task Types
            print("Loading Human PR Task Types...")
            ds_human_task = load_dataset("hao-li/AIDev", name="human_pr_task_type", split="train")
            self.data['human_task_types'] = ds_human_task.to_pandas()
            print(f"   Loaded {len(self.data['human_task_types'])} human PR task types")

            # 7. Load Repository data (for license filtering)
            print("Loading Repository metadata (for license filtering)...")
            ds_repo = load_dataset("hao-li/AIDev", name="repository", split="train")
            self.data['repositories'] = ds_repo.to_pandas()
            print(f"   Loaded {len(self.data['repositories'])} repositories")

            # 8. Load PR Comments (for non-author interaction filtering)
            print("Loading PR Comments (for interaction filtering)...")
            ds_pr_comments = load_dataset("hao-li/AIDev", name="pr_comments", split="train")
            self.data['pr_comments'] = ds_pr_comments.to_pandas()
            print(f"   Loaded {len(self.data['pr_comments'])} PR comments")

            # 9. Load PR Commits (for outlier filtering)
            print("Loading PR Commits (for outlier filtering)...")
            ds_commits = load_dataset("hao-li/AIDev", name="pr_commits", split="train")
            self.data['pr_commits'] = ds_commits.to_pandas()
            print(f"   Loaded {len(self.data['pr_commits'])} PR commits")

            # 10. Load PR Commit Details (for files changed filtering)
            print("Loading PR Commit Details (for files changed filtering)...")
            ds_commit_details = load_dataset("hao-li/AIDev", name="pr_commit_details", split="train")
            self.data['pr_commit_details'] = ds_commit_details.to_pandas()
            print(f"   Loaded {len(self.data['pr_commit_details'])} PR commit details")

            print(f"\nData Loading Complete:")
            print(f"  - AI PRs (100+ stars repos): {len(self.data['prs'])}")
            print(f"  - Human PRs (baseline): {len(self.data['human_prs'])}")
            print(f"  - Review Comments (inline): {len(self.data['comments'])}")
            print(f"  - Reviews (top-level): {len(self.data['reviews'])}")
            print(f"  - PR Task Types (AI): {len(self.data['task_types'])}")
            print(f"  - PR Task Types (Human): {len(self.data['human_task_types'])}")
            print(f"  - Repositories: {len(self.data['repositories'])}")
            print(f"  - PR Comments: {len(self.data['pr_comments'])}")
            print(f"  - PR Commits: {len(self.data['pr_commits'])}")
            print(f"  - PR Commit Details: {len(self.data['pr_commit_details'])}")

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
        df_human_prs = self.data['human_prs'].copy()
        df_comments = self.data['comments'].copy()
        df_reviews = self.data['reviews'].copy()
        df_task_types = self.data['task_types'].copy()
        df_human_task_types = self.data['human_task_types'].copy()

        print(f"\nInspecting columns for join keys...")
        print(f"  AI PRs columns: {list(df_prs.columns)[:15]}...")
        print(f"  Comments columns: {list(df_comments.columns)[:15]}...")
        print(f"  Reviews columns: {list(df_reviews.columns)[:15]}...")

        # === STEP 0: Concatenate AI PRs with Human PRs ===
        print("\n0. Concatenating AI PRs with Human PRs baseline...")
        print(f"   AI PRs: {len(df_prs)}")
        print(f"   Human PRs: {len(df_human_prs)}")

        # Ensure Human PRs have 'agent' column set to 'Human'
        df_human_prs['agent'] = 'Human'

        # Concatenate AI and Human PRs
        df_prs = pd.concat([df_prs, df_human_prs], ignore_index=True)
        print(f"   ✓ Combined PRs: {len(df_prs)}")

        # Concatenate task types
        df_task_types = pd.concat([df_task_types, df_human_task_types], ignore_index=True)
        print(f"   ✓ Combined Task Types: {len(df_task_types)}")

        # === STEP 0b: Apply Quality Filtering ===
        # Filter 1: Permissive licenses (MIT, Apache-2.0)
        # Filter 2: Meaningful human evaluation (non-author review/comment BEFORE closure)
        print("\n0b. Applying quality filtering...")
        print(f"   PRs before filtering: {len(df_prs)}")

        # Load additional data for filtering
        df_repositories = self.data['repositories'].copy()
        df_pr_comments = self.data['pr_comments'].copy()

        # Convert timestamps for filtering
        df_prs['created_at'] = pd.to_datetime(df_prs['created_at'], errors='coerce')
        df_prs['closed_at'] = pd.to_datetime(df_prs['closed_at'], errors='coerce')

        # === Filter 1: Permissive Licenses (MIT, Apache-2.0) ===
        prs_before = len(df_prs)
        allowed_licenses = ['MIT', 'Apache-2.0']
        allowed_repo_ids = df_repositories[df_repositories['license'].isin(allowed_licenses)]['id'].unique()
        df_prs = df_prs[df_prs['repo_id'].isin(allowed_repo_ids)]
        print(f"   ✓ Permissive licenses (MIT/Apache-2.0): {prs_before} → {len(df_prs)} (removed {prs_before - len(df_prs)})")

        # === Filter 2: Meaningful Human Evaluation ===
        # Keep only CLOSED PRs with at least one non-author review OR comment submitted BEFORE closure
        prs_before = len(df_prs)

        # First, filter to closed PRs only
        df_prs = df_prs[df_prs['state'] == 'closed']
        print(f"   ✓ Closed PRs only: {prs_before} → {len(df_prs)}")

        # Create PR info mapping: pr_id -> (user, closed_at)
        pr_info = df_prs[['id', 'user', 'closed_at']].copy()
        pr_info = pr_info.rename(columns={'id': 'pr_id', 'user': 'pr_user', 'closed_at': 'pr_closed_at'})

        # Process reviews: find PRs with non-author reviews submitted BEFORE closure
        prs_before = len(df_prs)
        df_reviews_copy = df_reviews.copy()
        df_reviews_copy['submitted_at'] = pd.to_datetime(df_reviews_copy['submitted_at'], errors='coerce')

        reviews_merged = df_reviews_copy.merge(pr_info, on='pr_id', how='inner')
        valid_reviews = reviews_merged[
            (reviews_merged['user'] != reviews_merged['pr_user']) &
            (reviews_merged['submitted_at'] < reviews_merged['pr_closed_at'])
        ]
        valid_review_pr_ids = set(valid_reviews['pr_id'].unique())
        print(f"      PRs with non-author review before closure: {len(valid_review_pr_ids)}")

        # Process comments: find PRs with non-author comments created BEFORE closure
        df_pr_comments_copy = df_pr_comments.copy()
        df_pr_comments_copy['created_at'] = pd.to_datetime(df_pr_comments_copy['created_at'], errors='coerce')

        comments_merged = df_pr_comments_copy.merge(pr_info, on='pr_id', how='inner')
        valid_comments = comments_merged[
            (comments_merged['user'] != comments_merged['pr_user']) &
            (comments_merged['created_at'] < comments_merged['pr_closed_at'])
        ]
        valid_comment_pr_ids = set(valid_comments['pr_id'].unique())
        print(f"      PRs with non-author comment before closure: {len(valid_comment_pr_ids)}")

        # Union: PR must have at least one valid review OR comment
        valid_pr_ids = valid_review_pr_ids | valid_comment_pr_ids
        df_prs = df_prs[df_prs['id'].isin(valid_pr_ids)]
        print(f"   ✓ Meaningful human evaluation (before closure): {prs_before} → {len(df_prs)} (removed {prs_before - len(df_prs)})")

        # Final summary
        print(f"\n   === Quality Filtering Complete ===")
        print(f"   Final PRs: {len(df_prs)}")
        print(f"   By agent:")
        for agent, count in df_prs['agent'].value_counts().items():
            print(f"      {agent}: {count}")

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

            # Filter out entries that are only code blocks (no meaningful text for sentiment)
            before_count = len(merged_comments)
            code_only_pattern = r'^(\[CODE_BLOCK\]|\[CODE\]|\s)+$'
            merged_comments = merged_comments[~merged_comments['clean_body'].str.match(code_only_pattern, na=False)]
            print(f"   ✓ Comments: Removed {before_count - len(merged_comments)} code-only entries")

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

            # Filter out entries that are only code blocks (no meaningful text for sentiment)
            before_count = len(merged_reviews)
            code_only_pattern = r'^(\[CODE_BLOCK\]|\[CODE\]|\s)+$'
            merged_reviews = merged_reviews[~merged_reviews['clean_body'].str.match(code_only_pattern, na=False)]
            print(f"   ✓ Reviews: Removed {before_count - len(merged_reviews)} code-only entries")

        # === STEP 8b: Language Detection (for multilingual sentiment analysis) ===
        print("\n8b. Detecting language for multilingual sentiment analysis...")

        def detect_lang_safe(text):
            """Detect language with fallback for errors."""
            try:
                if not text or len(text.strip()) < 5:
                    return "UNKNOWN"
                # Remove code placeholders for better detection
                clean_for_lang = text.replace('[CODE_BLOCK]', '').replace('[CODE]', '').strip()
                if len(clean_for_lang) < 5:
                    return "UNKNOWN"
                return detect_language(clean_for_lang)
            except Exception:
                return "UNKNOWN"

        # Detect language for comments
        print("   Detecting language in comments...")
        merged_comments['detected_lang'] = merged_comments['clean_body'].progress_apply(detect_lang_safe)
        lang_dist_comments = merged_comments['detected_lang'].value_counts()
        print(f"   ✓ Comments language distribution:")
        for lang, count in lang_dist_comments.head(10).items():
            pct = 100 * count / len(merged_comments)
            print(f"      {lang}: {count} ({pct:.1f}%)")

        # Detect language for reviews
        print("   Detecting language in reviews...")
        merged_reviews['detected_lang'] = merged_reviews['clean_body'].progress_apply(detect_lang_safe)
        lang_dist_reviews = merged_reviews['detected_lang'].value_counts()
        print(f"   ✓ Reviews language distribution:")
        for lang, count in lang_dist_reviews.head(10).items():
            pct = 100 * count / len(merged_reviews)
            print(f"      {lang}: {count} ({pct:.1f}%)")

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
        # Include merged_at for determining merge status in outcome analysis
        # Note: merged_at doesn't get _pr suffix because there's no collision with comments table
        common_cols = ['clean_body', 'agent', 'source', 'pr_type', 'created_at', 'closed_at', 'state', 'detected_lang', 'merged_at']

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
        Applica sentiment analysis multilingue con modelli specifici per lingua:
        - EN: cardiffnlp/twitter-roberta-base-sentiment-latest
        - FR: cmarkea/distilcamembert-base-sentiment
        - DE: oliverguhr/german-sentiment-bert
        - ES: pysentimiento/robertuito-sentiment-analysis
        - IT: MilaNLProc/feel-it-italian-sentiment
        - RU: blanchefort/rubert-base-cased-sentiment
        - ZH/JA/KO/altre: lxyuan/distilbert-base-multilingual-cased-sentiments-student
        - Fallback: cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual

        Friction Score = Probabilità Negativa.
        """
        print(">>> Phase 2: Running Multilingual Sentiment Analysis...")
        print(f"   Using device: {'GPU' if self.device_id == 0 else 'CPU'}")

        batch_size = 32
        loaded_pipelines = {}

        def get_or_load_pipeline(lang):
            """Lazy-load pipeline for specific language"""
            if lang in loaded_pipelines:
                return loaded_pipelines[lang], lang

            # Check if we have a specific model for this language
            if lang in self.sentiment_models_by_lang:
                config = self.sentiment_models_by_lang[lang]
            else:
                # Fallback to multilingual
                config = self.sentiment_models_by_lang["MULTILINGUAL"]
                lang = "MULTILINGUAL"

            if lang not in loaded_pipelines:
                print(f"   Loading {lang} model: {config['model_id']}...")
                try:
                    loaded_pipelines[lang] = pipeline(
                        "sentiment-analysis",
                        model=config['model_id'],
                        tokenizer=config['model_id'],
                        device=self.device_id,
                        top_k=None
                    )
                except Exception as e:
                    print(f"   ⚠️ Failed to load {lang} model: {e}, using multilingual fallback")
                    if "MULTILINGUAL" not in loaded_pipelines:
                        multi_config = self.sentiment_models_by_lang["MULTILINGUAL"]
                        loaded_pipelines["MULTILINGUAL"] = pipeline(
                            "sentiment-analysis",
                            model=multi_config['model_id'],
                            tokenizer=multi_config['model_id'],
                            device=self.device_id,
                            top_k=None
                        )
                    loaded_pipelines[lang] = loaded_pipelines["MULTILINGUAL"]
                    lang = "MULTILINGUAL"

            return loaded_pipelines[lang], lang

        def extract_sentiment_result(res, lang):
            """Extract friction score and label from model output, handling different label formats"""
            if res is None:
                return {'friction_score': 0.5, 'sentiment_label': 'neutral', 'model_used': 'fallback'}

            config = self.sentiment_models_by_lang.get(lang, self.sentiment_models_by_lang["MULTILINGUAL"])
            label_map = config.get('labels', {})

            # Handle 5-class models (like French) that need mapping to 3-class
            if config.get('map_to_3class', False):
                neg_score = 0
                pos_score = 0
                neu_score = 0
                for item in res:
                    mapped = label_map.get(item['label'], 'neutral')
                    if mapped == 'negative':
                        neg_score += item['score']
                    elif mapped == 'positive':
                        pos_score += item['score']
                    else:
                        neu_score += item['score']

                if neg_score > pos_score and neg_score > neu_score:
                    label = 'negative'
                elif pos_score > neg_score and pos_score > neu_score:
                    label = 'positive'
                else:
                    label = 'neutral'
                return {'friction_score': neg_score, 'sentiment_label': label, 'model_used': lang}

            # Handle 2-class models (like Italian) without neutral
            if config.get('no_neutral', False):
                neg_score = 0
                pos_score = 0
                for item in res:
                    mapped = label_map.get(item['label'], item['label'])
                    if mapped == 'negative':
                        neg_score = item['score']
                    elif mapped == 'positive':
                        pos_score = item['score']

                label = 'negative' if neg_score > pos_score else 'positive'
                return {'friction_score': neg_score, 'sentiment_label': label, 'model_used': lang}

            # Standard 3-class models
            neg_score = 0
            best_label = 'neutral'
            best_score = 0

            for item in res:
                mapped = label_map.get(item['label'], item['label'])
                if mapped == 'negative':
                    neg_score = item['score']
                if item['score'] > best_score:
                    best_score = item['score']
                    best_label = mapped

            return {'friction_score': neg_score, 'sentiment_label': best_label, 'model_used': lang}

        def run_sentiment_analysis_multilingual(df, name):
            """Helper to run multilingual sentiment analysis on a dataframe"""
            # Check if detected_lang column exists
            if 'detected_lang' not in df.columns:
                print(f"   ⚠️ No language column found, using English model for all")
                df = df.copy()
                df['detected_lang'] = 'EN'

            print(f"\n   Processing {len(df)} {name}:")

            # Get language distribution
            lang_dist = df['detected_lang'].value_counts()
            print(f"      Top languages: {dict(lang_dist.head(10))}")

            results_all = {}

            # Group by language and process each group
            for lang in df['detected_lang'].unique():
                df_lang = df[df['detected_lang'] == lang]
                if len(df_lang) == 0:
                    continue

                # Map UNKNOWN to EN
                actual_lang = 'EN' if lang == 'UNKNOWN' else lang

                pipe, used_lang = get_or_load_pipeline(actual_lang)
                texts = df_lang['clean_body'].tolist()

                results_lang = []
                max_len = 512 if used_lang == 'EN' else 256  # Shorter for multilingual models

                for i in tqdm(range(0, len(texts), batch_size), desc=f"   {name} ({lang}→{used_lang})", leave=False):
                    batch = texts[i:i+batch_size]
                    try:
                        preds = pipe(batch, truncation=True, max_length=max_len)
                        results_lang.extend(preds)
                    except Exception as e:
                        print(f"      Batch error ({lang}): {e}")
                        results_lang.extend([None] * len(batch))

                for idx, res in zip(df_lang.index, results_lang):
                    results_all[idx] = extract_sentiment_result(res, used_lang)

            # Reconstruct dataframe with results in original order
            df_copy = df.copy()
            df_copy['friction_score'] = df_copy.index.map(lambda x: results_all.get(x, {}).get('friction_score', 0.5))
            df_copy['sentiment_label'] = df_copy.index.map(lambda x: results_all.get(x, {}).get('sentiment_label', 'neutral'))
            df_copy['sentiment_model_used'] = df_copy.index.map(lambda x: results_all.get(x, {}).get('model_used', 'unknown'))
            df_copy['is_negative'] = df_copy['sentiment_label'] == 'negative'

            return df_copy

        # Analyze combined dataset
        print("\n   === Analyzing Combined Dataset ===")
        self.analyzed_df = run_sentiment_analysis_multilingual(self.dataset, "combined items")

        # Analyze comments separately
        print("\n   === Analyzing Comments ===")
        self.analyzed_comments_df = run_sentiment_analysis_multilingual(self.comments_df, "comments")

        # Analyze reviews separately
        print("\n   === Analyzing Reviews ===")
        self.analyzed_reviews_df = run_sentiment_analysis_multilingual(self.reviews_df, "reviews")

        # Clean up GPU memory
        for pipe in loaded_pipelines.values():
            del pipe
        loaded_pipelines.clear()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        # Summary statistics
        print("\n   === Sentiment Analysis Summary ===")
        print(f"   Combined: {len(self.analyzed_df)} items")
        print(f"      - Negative: {self.analyzed_df['is_negative'].sum()} ({100*self.analyzed_df['is_negative'].mean():.1f}%)")
        print(f"      - Mean friction: {self.analyzed_df['friction_score'].mean():.3f}")

        # Model usage statistics
        if 'sentiment_model_used' in self.analyzed_df.columns:
            model_usage = self.analyzed_df['sentiment_model_used'].value_counts()
            print(f"\n   Models used:")
            for model, count in model_usage.items():
                print(f"      {model}: {count} ({100*count/len(self.analyzed_df):.1f}%)")

        # Language breakdown
        if 'detected_lang' in self.analyzed_df.columns:
            print(f"\n   Friction by detected language (top 10):")
            for lang in self.analyzed_df['detected_lang'].value_counts().head(10).index:
                lang_data = self.analyzed_df[self.analyzed_df['detected_lang'] == lang]
                model = lang_data['sentiment_model_used'].iloc[0] if 'sentiment_model_used' in lang_data.columns else 'unknown'
                print(f"      {lang}: mean={lang_data['friction_score'].mean():.3f}, n={len(lang_data)}, model={model}")

        print(f"\n   Comments: {len(self.analyzed_comments_df)} items")
        print(f"      - Negative: {self.analyzed_comments_df['is_negative'].sum()} ({100*self.analyzed_comments_df['is_negative'].mean():.1f}%)")
        print(f"      - Mean friction: {self.analyzed_comments_df['friction_score'].mean():.3f}")

        print(f"\n   Reviews: {len(self.analyzed_reviews_df)} items")
        print(f"      - Negative: {self.analyzed_reviews_df['is_negative'].sum()} ({100*self.analyzed_reviews_df['is_negative'].mean():.1f}%)")
        print(f"      - Mean friction: {self.analyzed_reviews_df['friction_score'].mean():.3f}")

        print("\n✓ Multilingual Sentiment Analysis Complete.")

    # ==========================================
    # PHASE 2a: Multi-Model Sentiment Analysis
    # ==========================================
    def analyze_sentiment_multimodel(self):
        """
        Runs sentiment analysis with multiple models for robustness.

        This method:
        1. Analyzes text with 3 HuggingFace sentiment models (general-purpose)
        2. Analyzes text with SentiCR (code review-specific model)
        3. Calculates inter-model agreement (Cohen's Kappa)
        4. Creates ensemble predictions based on majority voting
        5. Reports confidence scores based on model consensus

        Models used:
        - twitter_roberta: Primary model, trained on tweets (3-class)
        - bertweet_sentiment: BERTweet trained on 850M tweets (3-class)
        - cardiffnlp_roberta: RoBERTa base for tweets (3-class)
        - SentiCR: Code review-specific model (Ahmed et al., 2017)

        The primary model (twitter_roberta) is used for the main friction_score.
        SentiCR provides domain-specific validation for code review context.

        References:
        - Landis & Koch (1977): κ interpretation (0.81-1.00 = almost perfect)
        - Ahmed et al. (2017): SentiCR for code review sentiment
        - arXiv:2401.10845: General models outperform SE-specific for sentiment
        """
        print("\n>>> Phase 2a: Multi-Model Sentiment Analysis (Robustness Check)...")
        print("   Running 4 sentiment models (3 general + 1 code review-specific)")

        print(f"   Using device: {'GPU' if self.device_id == 0 else 'CPU'}")

        # Sample for multi-model analysis (full dataset can be slow with 3 models)
        # Use stratified sample to maintain agent distribution
        df = self.analyzed_df.copy()

        # For efficiency, sample if dataset is large
        max_samples = 5000  # Analyze up to 5000 samples with all models
        if len(df) > max_samples:
            print(f"   Sampling {max_samples} items for multi-model analysis (stratified by agent)")
            df_sample = df.groupby('agent', group_keys=False).apply(
                lambda x: x.sample(min(len(x), int(max_samples * len(x) / len(df))), random_state=42)
            )
        else:
            df_sample = df

        print(f"   Analyzing {len(df_sample)} samples with 3 models...")

        texts = df_sample['clean_body'].tolist()
        batch_size = 32

        # Store predictions from each model
        model_predictions = {}

        for model_name, model_config in self.sentiment_models.items():
            print(f"\n   --- Model: {model_name} ---")
            print(f"   {model_config['description']}")

            # Check if model should be forced to CPU
            use_device = -1 if model_config.get('force_cpu', False) else self.device_id
            if model_config.get('force_cpu', False):
                print(f"   (Running on CPU due to known compatibility issues)")

            try:
                # Load model pipeline
                model_pipe = pipeline(
                    "sentiment-analysis",
                    model=model_config['model_id'],
                    tokenizer=model_config['model_id'],
                    device=use_device,
                    top_k=None
                )

                results = []
                for i in tqdm(range(0, len(texts), batch_size), desc=f"   {model_name}"):
                    batch = texts[i:i+batch_size]
                    try:
                        preds = model_pipe(batch, truncation=True, max_length=512)
                        results.extend(preds)
                    except Exception as e:
                        # Handle batch errors gracefully
                        print(f"      Batch error: {e}")
                        results.extend([None] * len(batch))

                # Extract predictions
                labels = []
                scores = []
                label_map = model_config['labels']

                for res in results:
                    if res is None:
                        labels.append('neutral')
                        scores.append(0.5)
                    else:
                        # Get the highest scoring label
                        top_pred = max(res, key=lambda x: x['score'])
                        raw_label = top_pred['label']

                        # Map to standardized labels
                        if raw_label.lower() in ['negative', 'neg']:
                            std_label = 'negative'
                        elif raw_label.lower() in ['positive', 'pos']:
                            std_label = 'positive'
                        elif raw_label in label_map:
                            std_label = label_map[raw_label]
                        else:
                            std_label = 'neutral'

                        labels.append(std_label)

                        # Get negative probability for friction score
                        neg_score = next(
                            (item['score'] for item in res
                             if item['label'].lower() in ['negative', 'neg'] or
                                (item['label'] in label_map and label_map[item['label']] == 'negative')),
                            0.0
                        )
                        scores.append(neg_score)

                model_predictions[model_name] = {
                    'labels': labels,
                    'neg_scores': scores,
                    'n_negative': sum(1 for l in labels if l == 'negative'),
                    'n_positive': sum(1 for l in labels if l == 'positive'),
                    'n_neutral': sum(1 for l in labels if l == 'neutral'),
                    'mean_neg_score': np.mean(scores)
                }

                print(f"      Negative: {model_predictions[model_name]['n_negative']} ({100*model_predictions[model_name]['n_negative']/len(labels):.1f}%)")
                print(f"      Mean neg score: {model_predictions[model_name]['mean_neg_score']:.3f}")

                # Clean up GPU memory
                del model_pipe
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass  # CUDA might be in corrupted state

            except Exception as e:
                print(f"      ⚠️ Model {model_name} failed on GPU: {e}")

                # Reset CUDA state after error (with protection)
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except:
                    print(f"      CUDA reset failed, continuing with CPU fallback...")

                # Retry on CPU as fallback
                print(f"      Retrying {model_name} on CPU...")
                try:
                    model_pipe_cpu = pipeline(
                        "sentiment-analysis",
                        model=model_config['model_id'],
                        tokenizer=model_config['model_id'],
                        device=-1,  # CPU
                        top_k=None
                    )

                    results = []
                    for i in tqdm(range(0, len(texts), batch_size), desc=f"   {model_name} (CPU)"):
                        batch = texts[i:i+batch_size]
                        try:
                            preds = model_pipe_cpu(batch, truncation=True, max_length=512)
                            results.extend(preds)
                        except Exception as batch_e:
                            print(f"      CPU Batch error: {batch_e}")
                            results.extend([None] * len(batch))

                    # Extract predictions (same logic as GPU)
                    labels = []
                    scores = []
                    label_map = model_config['labels']

                    for res in results:
                        if res is None:
                            labels.append('neutral')
                            scores.append(0.5)
                        else:
                            top_pred = max(res, key=lambda x: x['score'])
                            raw_label = top_pred['label']

                            if raw_label.lower() in ['negative', 'neg']:
                                std_label = 'negative'
                            elif raw_label.lower() in ['positive', 'pos']:
                                std_label = 'positive'
                            elif raw_label in label_map:
                                std_label = label_map[raw_label]
                            else:
                                std_label = 'neutral'

                            labels.append(std_label)
                            neg_score = next(
                                (item['score'] for item in res
                                 if item['label'].lower() in ['negative', 'neg'] or
                                    (item['label'] in label_map and label_map[item['label']] == 'negative')),
                                0.0
                            )
                            scores.append(neg_score)

                    model_predictions[model_name] = {
                        'labels': labels,
                        'neg_scores': scores,
                        'n_negative': sum(1 for l in labels if l == 'negative'),
                        'n_positive': sum(1 for l in labels if l == 'positive'),
                        'n_neutral': sum(1 for l in labels if l == 'neutral'),
                        'mean_neg_score': np.mean(scores)
                    }

                    print(f"      ✓ CPU fallback successful")
                    print(f"      Negative: {model_predictions[model_name]['n_negative']} ({100*model_predictions[model_name]['n_negative']/len(labels):.1f}%)")
                    print(f"      Mean neg score: {model_predictions[model_name]['mean_neg_score']:.3f}")

                    del model_pipe_cpu

                except Exception as cpu_e:
                    print(f"      ⚠️ CPU fallback also failed: {cpu_e}")
                    model_predictions[model_name] = None

        # === SentiCR: Code Review-Specific Model ===
        print(f"\n   --- Model: SentiCR (Code Review-Specific) ---")
        print("   Gradient Boosting model trained on 1600 code review comments (Ahmed et al., 2017)")
        print("   NOTE: SentiCR is BINARY (negative vs non-negative, no explicit positive class)")

        try:
            # Initialize SentiCR
            senticr = create_senticr_pipeline(algo="GBT")

            # Get predictions
            senticr_results = senticr.analyze(texts)

            # Extract labels and scores
            # SentiCR is binary: -1 (negative) and 0 (non-negative/neutral)
            senticr_labels = [r['label'] for r in senticr_results]
            senticr_scores = []

            # Calculate negative scores (using probability)
            for r in senticr_results:
                if r['polarity'] == -1:  # negative
                    senticr_scores.append(r['score'])
                else:  # non-negative (neutral)
                    # For non-negative, the neg_score is 1 - confidence
                    senticr_scores.append(1.0 - r['score'])

            model_predictions['senticr'] = {
                'labels': senticr_labels,
                'neg_scores': senticr_scores,
                'n_negative': sum(1 for l in senticr_labels if l == 'negative'),
                'n_positive': sum(1 for l in senticr_labels if l == 'positive'),
                'n_neutral': sum(1 for l in senticr_labels if l == 'neutral'),
                'mean_neg_score': np.mean(senticr_scores)
            }

            print(f"      Negative: {model_predictions['senticr']['n_negative']} ({100*model_predictions['senticr']['n_negative']/len(senticr_labels):.1f}%)")
            print(f"      Neutral: {model_predictions['senticr']['n_neutral']} ({100*model_predictions['senticr']['n_neutral']/len(senticr_labels):.1f}%)")
            print(f"      Positive: {model_predictions['senticr']['n_positive']} ({100*model_predictions['senticr']['n_positive']/len(senticr_labels):.1f}%)")
            print(f"      Mean neg score: {model_predictions['senticr']['mean_neg_score']:.3f}")

        except Exception as e:
            print(f"      ⚠️ SentiCR failed: {e}")
            import traceback
            traceback.print_exc()
            model_predictions['senticr'] = None

        # === Calculate Inter-Model Agreement ===
        print("\n   === Inter-Model Agreement Analysis ===")

        # Get model pairs with valid predictions
        valid_models = [m for m, p in model_predictions.items() if p is not None]

        if len(valid_models) < 2:
            print("   ⚠️ Not enough valid models for agreement analysis")
            return

        agreement_results = []

        # Pairwise Cohen's Kappa
        print("\n   Pairwise Cohen's Kappa (label agreement):")
        print("   " + "-" * 60)

        for i in range(len(valid_models)):
            for j in range(i + 1, len(valid_models)):
                m1, m2 = valid_models[i], valid_models[j]
                labels1 = model_predictions[m1]['labels']
                labels2 = model_predictions[m2]['labels']

                # Calculate Cohen's Kappa
                try:
                    kappa = cohen_kappa_score(labels1, labels2)

                    # Interpret kappa (Landis & Koch, 1977)
                    if kappa >= 0.81:
                        interp = "almost perfect"
                    elif kappa >= 0.61:
                        interp = "substantial"
                    elif kappa >= 0.41:
                        interp = "moderate"
                    elif kappa >= 0.21:
                        interp = "fair"
                    else:
                        interp = "slight/poor"

                    # Calculate simple agreement percentage
                    agreement_pct = sum(1 for a, b in zip(labels1, labels2) if a == b) / len(labels1) * 100

                    print(f"   {m1} vs {m2}: κ = {kappa:.3f} ({interp}), Agreement = {agreement_pct:.1f}%")

                    agreement_results.append({
                        'model_pair': f"{m1} vs {m2}",
                        'cohens_kappa': kappa,
                        'interpretation': interp,
                        'agreement_pct': agreement_pct
                    })

                except Exception as e:
                    print(f"   {m1} vs {m2}: Kappa calculation failed - {e}")

        # === Ensemble Predictions (Majority Voting) ===
        print("\n   === Ensemble Predictions (Majority Voting) ===")

        # Create ensemble label based on majority
        ensemble_labels = []
        ensemble_confidence = []

        for idx in range(len(texts)):
            votes = []
            for m in valid_models:
                if model_predictions[m] is not None:
                    votes.append(model_predictions[m]['labels'][idx])

            if votes:
                # Majority vote
                from collections import Counter
                vote_counts = Counter(votes)
                majority_label = vote_counts.most_common(1)[0][0]
                majority_count = vote_counts.most_common(1)[0][1]

                ensemble_labels.append(majority_label)
                ensemble_confidence.append(majority_count / len(votes))  # Confidence = proportion agreeing
            else:
                ensemble_labels.append('neutral')
                ensemble_confidence.append(0.0)

        # Summary statistics
        n_total = len(ensemble_labels)
        n_negative = sum(1 for l in ensemble_labels if l == 'negative')
        n_positive = sum(1 for l in ensemble_labels if l == 'positive')
        n_neutral = sum(1 for l in ensemble_labels if l == 'neutral')
        mean_confidence = np.mean(ensemble_confidence)

        print(f"   Ensemble results (n={n_total}):")
        print(f"      Negative: {n_negative} ({100*n_negative/n_total:.1f}%)")
        print(f"      Positive: {n_positive} ({100*n_positive/n_total:.1f}%)")
        print(f"      Neutral: {n_neutral} ({100*n_neutral/n_total:.1f}%)")
        print(f"      Mean confidence: {mean_confidence:.3f}")

        # High confidence predictions (all models agree)
        high_conf = sum(1 for c in ensemble_confidence if c == 1.0)
        print(f"      Unanimous agreement: {high_conf} ({100*high_conf/n_total:.1f}%)")

        # === Compare Primary Model vs Ensemble ===
        print("\n   === Primary Model vs Ensemble Comparison ===")
        primary_model = 'twitter_roberta'
        if primary_model in model_predictions and model_predictions[primary_model] is not None:
            primary_labels = model_predictions[primary_model]['labels']

            # Agreement between primary and ensemble
            primary_ensemble_agreement = sum(1 for a, b in zip(primary_labels, ensemble_labels) if a == b) / len(primary_labels) * 100
            primary_ensemble_kappa = cohen_kappa_score(primary_labels, ensemble_labels)

            print(f"   Primary ({primary_model}) vs Ensemble:")
            print(f"      Agreement: {primary_ensemble_agreement:.1f}%")
            print(f"      Cohen's κ: {primary_ensemble_kappa:.3f}")

            # Cases where ensemble disagrees with primary
            disagreements = [(i, primary_labels[i], ensemble_labels[i])
                           for i in range(len(primary_labels))
                           if primary_labels[i] != ensemble_labels[i]]
            print(f"      Disagreements: {len(disagreements)} cases")

        # === Store results ===
        self.results['multimodel_sentiment'] = {
            'models_used': valid_models,
            'n_samples': len(texts),
            'model_predictions': {m: {k: v for k, v in p.items() if k != 'labels' and k != 'neg_scores'}
                                 for m, p in model_predictions.items() if p is not None},
            'pairwise_agreement': agreement_results,
            'ensemble': {
                'n_negative': n_negative,
                'n_positive': n_positive,
                'n_neutral': n_neutral,
                'negative_rate': n_negative / n_total,
                'mean_confidence': mean_confidence,
                'unanimous_rate': high_conf / n_total
            }
        }

        # Add ensemble labels to the sample dataframe
        df_sample['ensemble_label'] = ensemble_labels
        df_sample['ensemble_confidence'] = ensemble_confidence

        # Store for later use
        self.multimodel_sample_df = df_sample

        print("\n   ✓ Multi-model sentiment analysis complete")
        print(f"   Mean inter-model κ: {np.mean([r['cohens_kappa'] for r in agreement_results]):.3f}")

    # ==========================================
    # PHASE 2b: Emotion Analysis (Ekman 7 Categories)
    # ==========================================
    def analyze_emotions(self):
        """
        Analyze emotions with multilingual support:
        - EN: j-hartmann/emotion-english-distilroberta-base (7 Ekman categories)
        - ES: pysentimiento/robertuito-emotion-analysis (Spanish emotions)
        - Other: MilaNLProc/xlm-emo-t (8 emotions, 19 languages)

        Categories normalized to: anger, disgust, fear, joy, sadness, surprise, neutral
        """
        print(">>> Phase 2b: Multilingual Emotion Analysis...")
        print(f"   Using device: {'GPU' if self.device_id == 0 else 'CPU'}")

        batch_size = 32
        loaded_emotion_pipes = {}

        # Unified emotion labels (normalized across all models)
        standard_emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

        def get_or_load_emotion_pipeline(lang):
            """Lazy-load emotion pipeline for specific language"""
            if lang in loaded_emotion_pipes:
                return loaded_emotion_pipes[lang], lang

            # Check for language-specific model
            if lang in self.emotion_models_by_lang:
                config = self.emotion_models_by_lang[lang]
            else:
                # Fallback to multilingual
                config = self.emotion_models_by_lang["MULTILINGUAL"]
                lang = "MULTILINGUAL"

            if lang not in loaded_emotion_pipes:
                print(f"   Loading {lang} emotion model: {config['model_id']}...")
                try:
                    loaded_emotion_pipes[lang] = pipeline(
                        "text-classification",
                        model=config['model_id'],
                        top_k=None,
                        device=self.device_id
                    )
                except Exception as e:
                    print(f"   ⚠️ Failed to load {lang} emotion model: {e}, using English fallback")
                    if "EN" not in loaded_emotion_pipes:
                        en_config = self.emotion_models_by_lang["EN"]
                        loaded_emotion_pipes["EN"] = pipeline(
                            "text-classification",
                            model=en_config['model_id'],
                            top_k=None,
                            device=self.device_id
                        )
                    loaded_emotion_pipes[lang] = loaded_emotion_pipes["EN"]
                    lang = "EN"

            return loaded_emotion_pipes[lang], lang

        def normalize_emotion_label(label, lang):
            """Normalize emotion labels from different models to standard categories"""
            label = label.lower()

            # Map Spanish emotions (pysentimiento)
            if lang == "ES":
                mapping = {
                    'others': 'neutral',
                    'joy': 'joy',
                    'sadness': 'sadness',
                    'anger': 'anger',
                    'surprise': 'surprise',
                    'disgust': 'disgust',
                    'fear': 'fear'
                }
                return mapping.get(label, 'neutral')

            # Map multilingual XLM-EMO-T emotions
            if lang == "MULTILINGUAL":
                mapping = {
                    'anger': 'anger',
                    'anticipation': 'neutral',  # No direct equivalent
                    'disgust': 'disgust',
                    'fear': 'fear',
                    'joy': 'joy',
                    'sadness': 'sadness',
                    'surprise': 'surprise',
                    'trust': 'neutral'  # No direct equivalent
                }
                return mapping.get(label, 'neutral')

            # English model already uses standard labels
            return label if label in standard_emotions else 'neutral'

        def process_emotion_result(res, lang):
            """Process emotion result and normalize to standard categories"""
            if res is None:
                return {'dominant': 'neutral', 'scores': {e: 0.0 for e in standard_emotions}}

            # Get dominant emotion
            dominant = max(res, key=lambda x: x['score'])
            dominant_normalized = normalize_emotion_label(dominant['label'], lang)

            # Aggregate scores to standard emotions
            emotion_scores = {e: 0.0 for e in standard_emotions}
            for item in res:
                normalized = normalize_emotion_label(item['label'], lang)
                if normalized in emotion_scores:
                    emotion_scores[normalized] = max(emotion_scores[normalized], item['score'])

            return {'dominant': dominant_normalized, 'scores': emotion_scores}

        # Process all texts grouped by language
        df = self.analyzed_df.copy()

        # Ensure detected_lang column exists
        if 'detected_lang' not in df.columns:
            print(f"   ⚠️ No language column found, using English model for all")
            df['detected_lang'] = 'EN'

        print(f"\n   Processing {len(df)} texts for emotion analysis...")
        lang_dist = df['detected_lang'].value_counts()
        print(f"      Top languages: {dict(lang_dist.head(10))}")

        # Initialize result columns
        all_results = {idx: None for idx in df.index}

        # Group by language and process
        for lang in df['detected_lang'].unique():
            df_lang = df[df['detected_lang'] == lang]
            if len(df_lang) == 0:
                continue

            actual_lang = 'EN' if lang == 'UNKNOWN' else lang
            pipe, used_lang = get_or_load_emotion_pipeline(actual_lang)
            texts = df_lang['clean_body'].tolist()

            max_len = 512 if used_lang == 'EN' else 256

            results_lang = []
            for i in tqdm(range(0, len(texts), batch_size), desc=f"   Emotions ({lang}→{used_lang})", leave=False):
                batch = texts[i:i+batch_size]
                try:
                    preds = pipe(batch, truncation=True, max_length=max_len)
                    results_lang.extend(preds)
                except Exception as e:
                    print(f"      Batch error ({lang}): {e}")
                    results_lang.extend([None] * len(batch))

            for idx, res in zip(df_lang.index, results_lang):
                all_results[idx] = process_emotion_result(res, used_lang)

        # Extract results and add to dataframe
        emotions = []
        emotion_scores = {e: [] for e in standard_emotions}

        for idx in df.index:
            result = all_results.get(idx)
            if result is None:
                result = {'dominant': 'neutral', 'scores': {e: 0.0 for e in standard_emotions}}

            emotions.append(result['dominant'])
            for e in standard_emotions:
                emotion_scores[e].append(result['scores'].get(e, 0.0))

        # Add to dataframe
        self.analyzed_df['dominant_emotion'] = emotions
        for emotion in standard_emotions:
            self.analyzed_df[f'emotion_{emotion}'] = emotion_scores[emotion]

        # Calculate aggregate negative emotion score
        self.analyzed_df['negative_emotion_score'] = (
            self.analyzed_df['emotion_anger'] +
            self.analyzed_df['emotion_disgust'] +
            self.analyzed_df['emotion_fear'] +
            self.analyzed_df['emotion_sadness']
        ) / 4

        # Clean up pipelines
        for pipe in loaded_emotion_pipes.values():
            del pipe
        loaded_emotion_pipes.clear()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        # Summary statistics
        print("\n   === Emotion Analysis Summary ===")
        emotion_dist = self.analyzed_df['dominant_emotion'].value_counts()
        for emotion, count in emotion_dist.items():
            pct = 100 * count / len(self.analyzed_df)
            print(f"      {emotion}: {count} ({pct:.1f}%)")

        # Emotion by agent
        print("\n   Negative emotion score by agent:")
        for agent in self.analyzed_df['agent'].unique():
            agent_df = self.analyzed_df[self.analyzed_df['agent'] == agent]
            mean_neg = agent_df['negative_emotion_score'].mean()
            print(f"      {agent}: {mean_neg:.4f}")

        print("\n✓ Multilingual Emotion Analysis Complete.")

    # ==========================================
    # PHASE 3: Topic Modeling (BERTopic) - Multilingual
    # ==========================================
    def extract_friction_topics(self):
        """
        Apply BERTopic on negative comments to identify friction causes.
        RQ3: Which specific topics generate the most friction?

        MULTILINGUAL APPROACH:
        - Groups negative comments by detected language
        - Uses language-specific embedding models for each group
        - Runs separate BERTopic for each language
        - Falls back to multilingual model for unsupported languages

        Uses best practices:
        - CountVectorizer with stopwords removal AFTER clustering
        - ClassTfidfTransformer to reduce frequent word impact
        - N-gram range (1,2) for better topic representations
        """
        print(">>> Phase 3: Multilingual Topic Modeling on Negative Comments...")

        from sklearn.feature_extraction.text import CountVectorizer
        from bertopic.vectorizers import ClassTfidfTransformer

        # Get negative comments with their indices and languages
        negative_mask = self.analyzed_df['is_negative']
        negative_df = self.analyzed_df[negative_mask].copy()

        if len(negative_df) < 10:
            print("   Not enough negative comments for Topic Modeling.")
            return

        print(f"   Processing {len(negative_df)} negative comments...")

        # Ensure language column exists
        if 'detected_lang' not in negative_df.columns:
            print("   ⚠️  No language column found. Running language detection...")
            negative_df['detected_lang'] = negative_df['clean_body'].progress_apply(
                lambda x: detect_language(str(x))[:2].upper() if pd.notna(x) and len(str(x).strip()) > 5 else "EN"
            )

        # Group by language
        lang_groups = negative_df.groupby('detected_lang')
        print(f"\n   Language distribution in negative comments:")
        for lang, group in lang_groups:
            print(f"      {lang}: {len(group)} comments")

        # Store results per language
        self.topic_models_by_lang = {}
        self.results['topics_by_lang'] = {}
        all_topics_info = []

        # Process each language group
        for lang, group_df in lang_groups:
            n_comments = len(group_df)

            # Skip very small groups
            if n_comments < 5:
                print(f"\n   ⏭️  Skipping {lang} (only {n_comments} comments, min=5)")
                continue

            print(f"\n   📊 Processing {lang} ({n_comments} comments)...")

            # Get language-specific embedding model
            if lang in self.embedding_models_by_lang:
                model_config = self.embedding_models_by_lang[lang]
            else:
                model_config = self.embedding_models_by_lang["MULTILINGUAL"]
                print(f"      Using multilingual fallback for {lang}")

            model_id = model_config["model_id"]
            print(f"      Loading embedding model: {model_id}")

            try:
                # Load language-specific embedding model
                embedding_model = SentenceTransformer(model_id, device=self.device)

                # Stopwords: use English for EN, None for others (let embeddings handle semantics)
                stopwords = "english" if lang == "EN" else None

                # CountVectorizer configuration
                vectorizer_model = CountVectorizer(
                    stop_words=stopwords,
                    min_df=2,
                    ngram_range=(1, 2)
                )

                # ClassTfidfTransformer to reduce impact of frequent words
                ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

                # Adjust min_topic_size based on group size
                min_topic_size = max(3, min(10, n_comments // 20))

                # Create and fit BERTopic model
                topic_model = BERTopic(
                    embedding_model=embedding_model,
                    vectorizer_model=vectorizer_model,
                    ctfidf_model=ctfidf_model,
                    min_topic_size=min_topic_size,
                    nr_topics="auto"
                )

                comments_list = group_df['clean_body'].tolist()
                topics, probs = topic_model.fit_transform(comments_list)

                # Extract topic info
                topic_info = topic_model.get_topic_info()
                topic_info['language'] = lang

                # Add to combined results
                all_topics_info.append(topic_info)

                # Store model and results for this language
                self.topic_models_by_lang[lang] = topic_model
                self.results['topics_by_lang'][lang] = topic_info

                # Print top topics for this language
                n_topics = len(topic_info) - 1  # Exclude outlier topic (-1)
                print(f"      ✓ Found {n_topics} topics")
                if n_topics > 0:
                    # Show top 3 topics (excluding outliers)
                    top_topics = topic_info[topic_info['Topic'] != -1].head(3)
                    for _, row in top_topics.iterrows():
                        print(f"         Topic {row['Topic']}: {row['Name'][:50]}...")

                # Free GPU memory
                del embedding_model
                torch.cuda.empty_cache() if self.device == "cuda" else None

            except Exception as e:
                print(f"      ❌ Error processing {lang}: {str(e)}")
                continue

        # Combine all topic info
        if all_topics_info:
            combined_topics = pd.concat(all_topics_info, ignore_index=True)
            self.results['topics'] = combined_topics
            print(f"\n   ✓ Total topics across all languages: {len(combined_topics)}")

            # Summary by language
            print("\n   📊 Topics Summary by Language:")
            for lang in self.results['topics_by_lang']:
                n_topics = len(self.results['topics_by_lang'][lang]) - 1  # Exclude outlier
                print(f"      {lang}: {n_topics} topics")

        # Keep reference to primary (English) model for backward compatibility
        if "EN" in self.topic_models_by_lang:
            self.topic_model = self.topic_models_by_lang["EN"]
        elif self.topic_models_by_lang:
            # Use the first available model
            first_lang = list(self.topic_models_by_lang.keys())[0]
            self.topic_model = self.topic_models_by_lang[first_lang]

        print("\n✓ Multilingual Topic Modeling Complete.")

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
        print(f"   Using device for classification: {'GPU' if self.device_id == 0 else 'CPU'}")

        # Try GPU first, fallback to CPU if needed
        current_device = self.device_id
        try:
            classifier = pipeline(
                "zero-shot-classification",
                model=self.models['category_classifier'],
                device=current_device
            )
        except Exception as e:
            print(f"   ⚠️ GPU initialization failed: {e}")
            print(f"   Falling back to CPU...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            current_device = -1
            classifier = pipeline(
                "zero-shot-classification",
                model=self.models['category_classifier'],
                device=-1
            )

        # Create hypothesis template for better accuracy
        hypothesis_template = "This code review comment is about {}."

        categories = []
        gpu_failed = False
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
                    if not gpu_failed and current_device == self.device_id:
                        print(f"\n   ⚠️ GPU classification failed: {e}")
                        print(f"   Switching to CPU...")
                        gpu_failed = True
                        del classifier
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        classifier = pipeline(
                            "zero-shot-classification",
                            model=self.models['category_classifier'],
                            device=-1
                        )
                        # Retry on CPU
                        try:
                            result = classifier(
                                text,
                                candidate_labels=labels,
                                hypothesis_template=hypothesis_template,
                                multi_label=False
                            )
                            categories.append(result['labels'][0])
                        except:
                            categories.append(self._classify_single_keyword(text))
                    else:
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
        Compares AI agents only (Human baseline excluded from statistical tests
        due to insufficient sample size in AIDev dataset).

        Statistical Methods:
        - Kruskal-Wallis H-test for omnibus comparison (non-parametric ANOVA)
        - Eta-squared (η²) for overall effect size
        - Dunn's test for post-hoc pairwise comparisons
          (IMPORTANT: Dunn's test is the proper post-hoc for KW, not Mann-Whitney U,
           because it uses the same shared rankings as Kruskal-Wallis)
        - Multiple comparison corrections:
          * Bonferroni: Conservative, controls FWER
          * Holm-Bonferroni: More powerful than Bonferroni, still controls FWER
          * Benjamini-Hochberg: Controls FDR, appropriate for exploratory analysis
        - Cliff's Delta for effect sizes (Romano et al., 2006 thresholds)

        References:
        - Dunn, O.J. (1964). Multiple comparisons using rank sums. Technometrics.
        - Romano, J. et al. (2006). Appropriate statistics for ordinal level data.
        """
        print(">>> Phase 4: Statistical Analysis...")
        df = self.analyzed_df

        # RQ1: Friction by Agent (including Human baseline for descriptive stats)
        friction_by_agent = df.groupby('agent')['friction_score'].agg(['mean', 'count', 'std']).reset_index()
        self.results['friction_stats'] = friction_by_agent

        print("\n   Friction Statistics by Agent (all agents):")
        print(friction_by_agent.to_string(index=False))

        # Check Human sample size and warn if too small
        human_count = friction_by_agent[friction_by_agent['agent'] == 'Human']['count'].values
        if len(human_count) > 0 and human_count[0] < 100:
            print(f"\n   ⚠️  WARNING: Human baseline has only {int(human_count[0])} samples.")
            print("   The AIDev dataset does not include review comments for Human PRs.")
            print("   Human will be EXCLUDED from statistical tests (insufficient power).")
            print("   Human data is retained for descriptive/exploratory purposes only.")

        # Filter out Human for statistical tests (insufficient sample size)
        df_stats = df[df['agent'] != 'Human'].copy()
        print(f"\n   Statistical analysis on {len(df_stats)} samples (AI agents only)")

        # RQ2: Statistical Test - Kruskal-Wallis across AI agents only
        print("\n   Kruskal-Wallis test across AI agents...")
        agent_groups = [group['friction_score'].values for name, group in df_stats.groupby('agent')]

        if len(agent_groups) >= 2 and all(len(g) >= 2 for g in agent_groups):
            try:
                stat, p_val = stats.kruskal(*agent_groups)

                # Calculate Eta-squared effect size for Kruskal-Wallis
                k = len(agent_groups)  # number of groups
                n = sum(len(g) for g in agent_groups)  # total observations
                eta_squared = (stat - k + 1) / (n - k) if (n - k) > 0 else 0

                # Interpret effect size
                if eta_squared >= 0.14:
                    effect_interp = "large"
                elif eta_squared >= 0.06:
                    effect_interp = "medium"
                else:
                    effect_interp = "small"

                self.results['kruskal_wallis_agents'] = {
                    'stat': stat,
                    'p_value': p_val,
                    'eta_squared': eta_squared,
                    'effect_size': effect_interp
                }
                sig = "Yes" if p_val < 0.05 else "No"
                print(f"   Kruskal-Wallis test: H={stat:.2f}, p={p_val:.4f}, Significant: {sig}")
                print(f"   Effect size: η²={eta_squared:.4f} ({effect_interp})")

                # Post-hoc pairwise comparisons using DUNN'S TEST
                # ============================================================
                # METHODOLOGICAL NOTE:
                # Dunn's test is the appropriate post-hoc test after Kruskal-Wallis because:
                # 1. It uses the SAME shared rankings calculated by Kruskal-Wallis
                # 2. Mann-Whitney U re-ranks only two groups at a time (different data!)
                # 3. Dunn's test uses the pooled variance implied by KW null hypothesis
                # Reference: Dunn, O.J. (1964). Multiple comparisons using rank sums.
                # ============================================================
                if p_val < 0.05:
                    print("\n   Post-hoc: Dunn's Test (proper post-hoc for Kruskal-Wallis)")
                    print("   NOTE: Dunn's test uses same rankings as KW, unlike Mann-Whitney U")

                    agents = list(df_stats['agent'].unique())
                    n_comparisons = len(agents) * (len(agents) - 1) // 2
                    print(f"   Number of pairwise comparisons: {n_comparisons}")

                    # Run Dunn's test using scikit-posthocs
                    # Returns a symmetric matrix of p-values
                    dunn_results = sp.posthoc_dunn(
                        df_stats,
                        val_col='friction_score',
                        group_col='agent',
                        p_adjust=None  # We'll apply our own corrections
                    )

                    # Extract pairwise p-values
                    pairwise_results = []
                    raw_p_values = []
                    pair_names = []

                    for i in range(len(agents)):
                        for j in range(i + 1, len(agents)):
                            a1, a2 = agents[i], agents[j]
                            p_raw = dunn_results.loc[a1, a2]
                            raw_p_values.append(p_raw)
                            pair_names.append(f"{a1} vs {a2}")

                    # Apply multiple comparison corrections
                    # ============================================================
                    # CORRECTION METHODS:
                    # 1. Bonferroni: Conservative, controls FWER, p_adj = p * k
                    # 2. Holm-Bonferroni: Uniformly more powerful than Bonferroni,
                    #    step-down procedure, still controls FWER
                    # 3. Benjamini-Hochberg (FDR): Controls False Discovery Rate,
                    #    more powerful, appropriate for exploratory research
                    # ============================================================
                    _, p_bonferroni, _, _ = multipletests(raw_p_values, method='bonferroni')
                    _, p_holm, _, _ = multipletests(raw_p_values, method='holm')
                    _, p_bh, _, _ = multipletests(raw_p_values, method='fdr_bh')

                    print("\n   Multiple comparison corrections applied:")
                    print("   - Bonferroni: Controls FWER (conservative)")
                    print("   - Holm: Controls FWER (uniformly more powerful than Bonferroni)")
                    print("   - Benjamini-Hochberg: Controls FDR (more powerful, for exploration)")

                    # Calculate Cliff's Delta for effect sizes
                    idx = 0
                    for i in range(len(agents)):
                        for j in range(i + 1, len(agents)):
                            a1, a2 = agents[i], agents[j]
                            g1 = df_stats[df_stats['agent'] == a1]['friction_score'].values
                            g2 = df_stats[df_stats['agent'] == a2]['friction_score'].values

                            # Cliff's Delta calculation
                            # δ = (# of times g1 > g2 - # of times g1 < g2) / (n1 * n2)
                            n1, n2 = len(g1), len(g2)
                            greater = sum(1 for x in g1 for y in g2 if x > y)
                            less = sum(1 for x in g1 for y in g2 if x < y)
                            cliff_delta = (greater - less) / (n1 * n2)

                            # Interpret Cliff's Delta using ROMANO 2006 thresholds
                            # ============================================================
                            # CORRECTED THRESHOLDS (Romano et al., 2006):
                            # |δ| < 0.147  → negligible
                            # |δ| < 0.33   → small
                            # |δ| < 0.474  → medium
                            # |δ| >= 0.474 → large
                            # Reference: Romano, J. et al. (2006). Appropriate statistics
                            # for ordinal level data. Florida Association of IR.
                            # ============================================================
                            abs_delta = abs(cliff_delta)
                            if abs_delta >= 0.474:
                                delta_interp = "large"
                            elif abs_delta >= 0.33:
                                delta_interp = "medium"
                            elif abs_delta >= 0.147:
                                delta_interp = "small"
                            else:
                                delta_interp = "negligible"

                            pairwise_results.append({
                                'pair': pair_names[idx],
                                'n1': n1,
                                'n2': n2,
                                'p_dunn_raw': raw_p_values[idx],
                                'p_bonferroni': p_bonferroni[idx],
                                'p_holm': p_holm[idx],
                                'p_bh': p_bh[idx],
                                'cliff_delta': cliff_delta,
                                'effect_size': delta_interp,
                                'significant_raw': raw_p_values[idx] < 0.05,
                                'significant_bonferroni': p_bonferroni[idx] < 0.05,
                                'significant_holm': p_holm[idx] < 0.05,
                                'significant_bh': p_bh[idx] < 0.05
                            })
                            idx += 1

                    self.results['pairwise_tests'] = pairwise_results

                    # Print summary
                    print("\n   Pairwise Dunn's Test Results:")
                    print("   " + "-" * 95)
                    print(f"   {'Pair':<25} {'p_raw':<10} {'p_Bonf':<10} {'p_Holm':<10} {'p_BH':<10} {'δ':<8} {'Effect':<12}")
                    print("   " + "-" * 95)
                    for r in pairwise_results:
                        sig = "***" if r['significant_bonferroni'] else ("**" if r['significant_holm'] else ("*" if r['significant_bh'] else ""))
                        print(f"   {r['pair']:<25} {r['p_dunn_raw']:<10.4f} {r['p_bonferroni']:<10.4f} {r['p_holm']:<10.4f} {r['p_bh']:<10.4f} {r['cliff_delta']:<8.3f} {r['effect_size']:<12} {sig}")

                    print("\n   Legend: *** p<0.05 (Bonferroni), ** p<0.05 (Holm), * p<0.05 (BH/FDR)")
            except Exception as e:
                print(f"   Kruskal-Wallis test failed: {e}")
        else:
            print("   Insufficient data for Kruskal-Wallis test")
        
        # RQ4: Correlation with Merge Outcome
        # Use merged_at column (more reliable than state for determining merge status)
        print("\n   RQ4: Correlation Friction <-> Merge Success...")

        merge_col = None
        for possible_name in ['merged_at', 'merged_at_pr', 'pr_merged_at']:
            if possible_name in df.columns:
                merge_col = possible_name
                break

        # Fallback to state column if merged_at not available
        state_col = None
        if merge_col is None:
            for possible_name in ['state', 'state_pr', 'status']:
                if possible_name in df.columns:
                    state_col = possible_name
                    break

        if merge_col:
            # Use merged_at: if not null, PR was merged
            df['is_merged'] = df[merge_col].notna().astype(int)
            print(f"   Using '{merge_col}' column to determine merge status")
        elif state_col:
            # Fallback: check state column for 'merged' keyword
            df['is_merged'] = df[state_col].apply(
                lambda x: 1 if 'merged' in str(x).lower() else 0
            )
            print(f"   Using '{state_col}' column to determine merge status (fallback)")
        else:
            print("   Warning: No merged_at or state column found for correlation analysis")
            df['is_merged'] = None

        if df['is_merged'] is not None and df['is_merged'].notna().any():
            # Check if there's variance in is_merged (required for correlation)
            merge_variance = df['is_merged'].nunique()
            if merge_variance < 2:
                print(f"   Warning: is_merged has no variance (all values = {df['is_merged'].iloc[0]})")
                print("   Cannot calculate Point-Biserial correlation without variance in both variables")
            else:
                # Calculate Point-biserial correlation
                valid_mask = df['is_merged'].notna() & df['friction_score'].notna()
                if valid_mask.sum() >= 10:
                    corr, p_val_corr = stats.pointbiserialr(
                        df.loc[valid_mask, 'friction_score'],
                        df.loc[valid_mask, 'is_merged']
                    )
                    self.results['correlation'] = {'r': corr, 'p': p_val_corr}
                    print(f"   Point-Biserial Correlation: r = {corr:.3f}, p = {p_val_corr:.4f}")

                    # Interpret correlation
                    if abs(corr) < 0.1:
                        interp = "negligible"
                    elif abs(corr) < 0.3:
                        interp = "small"
                    elif abs(corr) < 0.5:
                        interp = "medium"
                    else:
                        interp = "large"
                    print(f"   Effect size: {interp}")
                else:
                    print("   Warning: Not enough valid data for correlation analysis")

            # Merge rate by agent (use full df for descriptive stats)
            merge_rates = df.groupby('agent')['is_merged'].agg(['mean', 'count']).reset_index()
            merge_rates.columns = ['agent', 'merge_rate', 'count']
            self.results['merge_rates'] = merge_rates
            print("\n   Merge rates by agent:")
            for _, row in merge_rates.iterrows():
                print(f"      {row['agent']}: {row['merge_rate']*100:.1f}% (n={row['count']})")

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
    # CONFOUNDING VARIABLE ANALYSIS
    # ==========================================
    def analyze_confounders(self):
        """
        Controls for confounding variables using multiple regression.

        Potential confounders in code review friction:
        - PR type (fix, feat, docs, etc.) - different PR types may receive different tones
        - PR duration - longer PRs may accumulate more friction
        - Source type (comment vs review) - inline comments are more critical
        - Repository characteristics - different projects have different cultures

        This analysis answers: "After controlling for these factors,
        do AI agents still differ significantly in friction?"

        Methods:
        1. OLS regression with agent as categorical predictor + confounders
        2. Report adjusted R² and coefficient significance
        3. Compare unadjusted vs adjusted agent effects

        References:
        - PMC4017459: How to control confounding effects by statistical analysis
        """
        print("\n>>> Confounding Variable Analysis...")
        print("   Controlling for: PR type, source type, PR duration")

        df = self.analyzed_df.copy()

        # === Prepare confounding variables ===
        confounders_available = []

        # 1. PR Type (categorical confounder)
        if 'pr_type' in df.columns:
            df['pr_type_clean'] = df['pr_type'].fillna('unknown')
            # Keep only top categories to avoid sparse encoding
            top_types = df['pr_type_clean'].value_counts().head(6).index.tolist()
            df['pr_type_grouped'] = df['pr_type_clean'].apply(
                lambda x: x if x in top_types else 'other'
            )
            confounders_available.append('pr_type_grouped')
            print(f"   ✓ PR type available ({len(top_types)} categories)")

        # 2. Source type (comment vs review)
        if 'source' in df.columns:
            df['is_comment'] = (df['source'] == 'comment').astype(int)
            confounders_available.append('is_comment')
            print("   ✓ Source type available (comment vs review)")

        # 3. PR duration (continuous confounder)
        if 'pr_duration_seconds' in df.columns:
            # Log transform to handle skewness, add 1 to avoid log(0)
            df['log_pr_duration'] = np.log1p(df['pr_duration_seconds'].fillna(0))
            confounders_available.append('log_pr_duration')
            print("   ✓ PR duration available (log-transformed)")

        # 4. Comment/review length as proxy for complexity
        if 'clean_body' in df.columns:
            df['text_length'] = df['clean_body'].str.len().fillna(0)
            df['log_text_length'] = np.log1p(df['text_length'])
            confounders_available.append('log_text_length')
            print("   ✓ Text length available (log-transformed)")

        if len(confounders_available) == 0:
            print("   ⚠️  No confounding variables available. Skipping analysis.")
            return

        print(f"\n   Building regression model with {len(confounders_available)} confounders...")

        # === Encode agent as dummy variables ===
        # Use Human as reference category if available, otherwise use the agent with lowest friction
        agents = df['agent'].unique()
        if 'Human' in agents:
            reference_agent = 'Human'
        else:
            # Use OpenAI_Codex as reference (lowest friction)
            reference_agent = 'OpenAI_Codex' if 'OpenAI_Codex' in agents else agents[0]

        print(f"   Reference agent: {reference_agent}")

        # Create dummy variables for agents (excluding reference)
        agent_dummies = pd.get_dummies(df['agent'], prefix='agent', drop_first=False)
        # Drop reference category
        if f'agent_{reference_agent}' in agent_dummies.columns:
            agent_dummies = agent_dummies.drop(columns=[f'agent_{reference_agent}'])

        # === Create dummy variables for categorical confounders ===
        confounder_dummies = pd.DataFrame()

        if 'pr_type_grouped' in confounders_available:
            pr_type_dummies = pd.get_dummies(df['pr_type_grouped'], prefix='prtype', drop_first=True)
            confounder_dummies = pd.concat([confounder_dummies, pr_type_dummies], axis=1)

        # === Build design matrix ===
        X_confounders = pd.DataFrame()

        # Add continuous confounders
        for conf in confounders_available:
            if conf in ['is_comment', 'log_pr_duration', 'log_text_length']:
                X_confounders[conf] = df[conf]

        # Add categorical confounder dummies
        if len(confounder_dummies) > 0:
            X_confounders = pd.concat([X_confounders, confounder_dummies], axis=1)

        # Combine: agents + confounders
        X_full = pd.concat([agent_dummies, X_confounders], axis=1)
        X_full = X_full.fillna(0)

        # Ensure all columns are numeric (fix "Pandas data cast to numpy dtype of object" error)
        for col in X_full.columns:
            X_full[col] = pd.to_numeric(X_full[col], errors='coerce').fillna(0)

        # Add constant for intercept
        X_full = add_constant(X_full)

        # Dependent variable - ensure numeric
        y = pd.to_numeric(df['friction_score'], errors='coerce')

        # Remove rows with NaN
        valid_idx = ~(X_full.isna().any(axis=1) | y.isna())
        X_full = X_full[valid_idx].astype(float)
        y = y[valid_idx].astype(float)

        print(f"   Sample size for regression: n={len(y)}")

        # === Model 1: Agents only (unadjusted) ===
        print("\n   --- Model 1: Unadjusted (Agents Only) ---")
        X_agents_only = agent_dummies[valid_idx].copy().astype(float)
        X_agents_only = add_constant(X_agents_only)
        try:
            model_unadjusted = OLS(y, X_agents_only).fit()
            print(f"   R² = {model_unadjusted.rsquared:.4f}")
            print(f"   Adjusted R² = {model_unadjusted.rsquared_adj:.4f}")

            # Store unadjusted coefficients
            unadjusted_coefs = {}
            for col in agent_dummies.columns:
                if col in model_unadjusted.params:
                    coef = model_unadjusted.params[col]
                    pval = model_unadjusted.pvalues[col]
                    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                    unadjusted_coefs[col] = {'coef': coef, 'pval': pval}
                    agent_name = col.replace('agent_', '')
                    print(f"      {agent_name}: β={coef:.4f}, p={pval:.4f} {sig}")
        except Exception as e:
            print(f"   ⚠️  Unadjusted model failed: {e}")
            model_unadjusted = None

        # === Model 2: Agents + Confounders (adjusted) ===
        print("\n   --- Model 2: Adjusted (Agents + Confounders) ---")
        try:
            model_adjusted = OLS(y, X_full).fit()
            print(f"   R² = {model_adjusted.rsquared:.4f}")
            print(f"   Adjusted R² = {model_adjusted.rsquared_adj:.4f}")

            # Store adjusted coefficients
            adjusted_coefs = {}
            print("\n   Agent effects (controlling for confounders):")
            for col in agent_dummies.columns:
                if col in model_adjusted.params:
                    coef = model_adjusted.params[col]
                    pval = model_adjusted.pvalues[col]
                    ci_low, ci_high = model_adjusted.conf_int().loc[col]
                    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                    adjusted_coefs[col] = {'coef': coef, 'pval': pval, 'ci_low': ci_low, 'ci_high': ci_high}
                    agent_name = col.replace('agent_', '')
                    print(f"      {agent_name}: β={coef:.4f} [{ci_low:.4f}, {ci_high:.4f}], p={pval:.4f} {sig}")

            print("\n   Confounder effects:")
            for col in X_confounders.columns:
                if col in model_adjusted.params:
                    coef = model_adjusted.params[col]
                    pval = model_adjusted.pvalues[col]
                    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                    print(f"      {col}: β={coef:.4f}, p={pval:.4f} {sig}")

            # === Compare unadjusted vs adjusted ===
            if model_unadjusted is not None:
                print("\n   --- Coefficient Change (Unadjusted → Adjusted) ---")
                for col in agent_dummies.columns:
                    if col in unadjusted_coefs and col in adjusted_coefs:
                        unadj = unadjusted_coefs[col]['coef']
                        adj = adjusted_coefs[col]['coef']
                        change_pct = ((adj - unadj) / abs(unadj)) * 100 if unadj != 0 else 0
                        agent_name = col.replace('agent_', '')
                        print(f"      {agent_name}: {unadj:.4f} → {adj:.4f} ({change_pct:+.1f}%)")

            # Store results
            self.results['confounder_analysis'] = {
                'model_unadjusted': {
                    'rsquared': model_unadjusted.rsquared if model_unadjusted else None,
                    'rsquared_adj': model_unadjusted.rsquared_adj if model_unadjusted else None,
                    'coefficients': unadjusted_coefs if model_unadjusted else {}
                },
                'model_adjusted': {
                    'rsquared': model_adjusted.rsquared,
                    'rsquared_adj': model_adjusted.rsquared_adj,
                    'coefficients': adjusted_coefs,
                    'n_confounders': len(confounders_available)
                },
                'confounders_used': confounders_available,
                'reference_agent': reference_agent,
                'n_observations': len(y)
            }

            print(f"\n   ✓ Confounder analysis complete")
            print(f"   Interpretation: Coefficients represent difference from {reference_agent} baseline")
            print(f"   Note: Positive β means MORE friction than {reference_agent}")

        except Exception as e:
            print(f"   ⚠️  Adjusted model failed: {e}")
            import traceback
            traceback.print_exc()

    # ==========================================
    # STATISTICAL POWER ANALYSIS
    # ==========================================
    def analyze_power(self):
        """
        Performs statistical power analysis for the main comparisons.

        Power analysis answers:
        1. What is the achieved power given our sample size and observed effect?
        2. What sample size would be needed to detect smaller effects?

        Key concepts:
        - Power (1-β): Probability of detecting a true effect (target: 0.80)
        - α: Significance level (0.05)
        - Effect size: Magnitude of the difference (Cohen's d or η²)

        References:
        - Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
        - PMC8441096: Sample size determination and power analysis using G*Power
        """
        print("\n>>> Statistical Power Analysis...")

        df = self.analyzed_df

        # === 1. Post-hoc power for Kruskal-Wallis (omnibus test) ===
        print("\n   --- Power Analysis for Kruskal-Wallis Test ---")

        if 'kruskal_wallis_agents' in self.results:
            kw_result = self.results['kruskal_wallis_agents']
            eta_sq = kw_result.get('eta_squared', 0)

            # Convert η² to Cohen's f for ANOVA power analysis
            # f = sqrt(η² / (1 - η²))
            if eta_sq > 0 and eta_sq < 1:
                cohens_f = np.sqrt(eta_sq / (1 - eta_sq))
            else:
                cohens_f = 0.1  # default small effect

            n_groups = len(df['agent'].unique())
            n_total = len(df)
            n_per_group = n_total / n_groups  # average

            print(f"   Observed effect size: η² = {eta_sq:.4f}")
            print(f"   Cohen's f = {cohens_f:.4f}")
            print(f"   Number of groups: {n_groups}")
            print(f"   Total N: {n_total}")
            print(f"   Average n per group: {n_per_group:.0f}")

            # Calculate achieved power using F-test approximation
            try:
                power_analysis = FTestAnovaPower()
                achieved_power = power_analysis.solve_power(
                    effect_size=cohens_f,
                    nobs=n_total,
                    alpha=0.05,
                    k_groups=n_groups
                )
                print(f"\n   Achieved statistical power: {achieved_power:.4f} ({achieved_power*100:.1f}%)")

                if achieved_power >= 0.80:
                    print("   ✓ Power is adequate (≥ 0.80)")
                else:
                    print("   ⚠️  Power is below recommended threshold (< 0.80)")

                    # Calculate required N for 80% power
                    required_n = power_analysis.solve_power(
                        effect_size=cohens_f,
                        power=0.80,
                        alpha=0.05,
                        k_groups=n_groups
                    )
                    print(f"   Required N for 80% power: {required_n:.0f}")

                self.results['power_analysis_omnibus'] = {
                    'test': 'Kruskal-Wallis (ANOVA approximation)',
                    'eta_squared': eta_sq,
                    'cohens_f': cohens_f,
                    'n_groups': n_groups,
                    'n_total': n_total,
                    'achieved_power': achieved_power,
                    'alpha': 0.05
                }

            except Exception as e:
                print(f"   ⚠️  Power calculation failed: {e}")

        # === 2. Post-hoc power for pairwise comparisons ===
        print("\n   --- Power Analysis for Pairwise Comparisons ---")

        if 'pairwise_tests' in self.results:
            pairwise_power = []
            power_calc = TTestIndPower()

            print(f"   {'Comparison':<30} {'n1':<8} {'n2':<8} {'δ':<10} {'d':<10} {'Power':<10}")
            print("   " + "-" * 76)

            for test in self.results['pairwise_tests']:
                pair = test['pair']
                n1 = test['n1']
                n2 = test['n2']
                cliff_d = test['cliff_delta']

                # Convert Cliff's Delta to Cohen's d (approximation)
                # Cohen's d ≈ Cliff's δ * π / √3 ≈ δ * 1.81
                # More accurate: d = 2 * δ / √(1 - δ²)
                if abs(cliff_d) < 0.99:
                    cohens_d = 2 * cliff_d / np.sqrt(1 - cliff_d**2)
                else:
                    cohens_d = cliff_d * 1.81  # fallback

                try:
                    # Calculate power for two-sample t-test (approximation for Mann-Whitney)
                    # Use harmonic mean of sample sizes for unequal groups
                    n_harmonic = 2 * n1 * n2 / (n1 + n2)

                    achieved_power = power_calc.solve_power(
                        effect_size=abs(cohens_d),
                        nobs1=n1,
                        ratio=n2/n1,
                        alpha=0.05,
                        alternative='two-sided'
                    )

                    power_status = "✓" if achieved_power >= 0.80 else "⚠️"
                    print(f"   {pair:<30} {n1:<8} {n2:<8} {cliff_d:<10.3f} {cohens_d:<10.3f} {achieved_power:<10.3f} {power_status}")

                    pairwise_power.append({
                        'pair': pair,
                        'n1': n1,
                        'n2': n2,
                        'cliff_delta': cliff_d,
                        'cohens_d': cohens_d,
                        'achieved_power': achieved_power,
                        'adequate_power': achieved_power >= 0.80
                    })

                except Exception as e:
                    print(f"   {pair:<30} Power calculation failed: {e}")

            self.results['power_analysis_pairwise'] = pairwise_power

            # Summary
            adequate_count = sum(1 for p in pairwise_power if p['adequate_power'])
            total_count = len(pairwise_power)
            print(f"\n   Summary: {adequate_count}/{total_count} comparisons have adequate power (≥ 0.80)")

            # Identify underpowered comparisons
            underpowered = [p for p in pairwise_power if not p['adequate_power']]
            if underpowered:
                print("\n   Underpowered comparisons (may miss true effects):")
                for p in underpowered:
                    print(f"      - {p['pair']}: power = {p['achieved_power']:.3f}")

        # === 3. Sensitivity analysis: Minimum detectable effect size ===
        print("\n   --- Sensitivity Analysis ---")
        print("   Minimum detectable effect sizes at 80% power:")

        # For the smallest group comparison
        if 'pairwise_tests' in self.results:
            # Find comparison with smallest sample
            min_comparison = min(self.results['pairwise_tests'],
                               key=lambda x: min(x['n1'], x['n2']))
            n_small = min(min_comparison['n1'], min_comparison['n2'])
            n_large = max(min_comparison['n1'], min_comparison['n2'])

            try:
                min_effect = power_calc.solve_power(
                    nobs1=n_small,
                    ratio=n_large/n_small,
                    power=0.80,
                    alpha=0.05,
                    alternative='two-sided'
                )
                print(f"   Smallest comparison ({min_comparison['pair']}): Cohen's d ≥ {min_effect:.3f}")

                # Convert to Cliff's Delta
                min_cliff = min_effect / np.sqrt(4 + min_effect**2)
                print(f"   (Equivalent Cliff's δ ≥ {min_cliff:.3f})")

                self.results['power_sensitivity'] = {
                    'smallest_comparison': min_comparison['pair'],
                    'n_small': n_small,
                    'n_large': n_large,
                    'min_detectable_cohens_d': min_effect,
                    'min_detectable_cliff_delta': min_cliff
                }

            except Exception as e:
                print(f"   Sensitivity calculation failed: {e}")

        print("\n   ✓ Power analysis complete")
        print("   Note: Power calculations use t-test/ANOVA approximations for non-parametric tests")

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
    # PHASE 5a: Emotion Visualizations
    # ==========================================
    def visualize_emotions(self):
        """
        Generate visualizations for Ekman emotion analysis.
        """
        print(">>> Phase 5a: Emotion Visualizations...")
        plots_dir = os.path.join(self.run_dir, "plots")
        emotions_dir = os.path.join(plots_dir, "emotions")
        os.makedirs(emotions_dir, exist_ok=True)

        df = self.analyzed_df

        if 'dominant_emotion' not in df.columns:
            print("   No emotion data found. Skipping emotion visualizations.")
            return

        # 1. Dominant Emotion Distribution
        plt.figure(figsize=(12, 7))
        emotion_order = ['anger', 'disgust', 'fear', 'sadness', 'neutral', 'surprise', 'joy']
        emotion_counts = df['dominant_emotion'].value_counts()
        colors = {'anger': 'red', 'disgust': 'purple', 'fear': 'orange',
                  'sadness': 'blue', 'neutral': 'gray', 'surprise': 'yellow', 'joy': 'green'}
        bar_colors = [colors.get(e, 'gray') for e in emotion_counts.index]
        plt.bar(emotion_counts.index, emotion_counts.values, color=bar_colors, edgecolor='black')
        plt.title("Dominant Emotion Distribution (Ekman Categories)", fontsize=14, fontweight='bold')
        plt.xlabel("Emotion", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(emotions_dir, "emotion_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 2. Emotion by Agent (Stacked Bar)
        plt.figure(figsize=(14, 8))
        emotion_agent = df.groupby(['agent', 'dominant_emotion']).size().unstack(fill_value=0)
        emotion_agent_pct = emotion_agent.div(emotion_agent.sum(axis=1), axis=0) * 100
        emotion_agent_pct.plot(kind='bar', stacked=True, colormap='Set3', figsize=(14, 8))
        plt.title("Emotion Distribution by Agent", fontsize=14, fontweight='bold')
        plt.xlabel("Agent", fontsize=12)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(emotions_dir, "emotion_by_agent_stacked.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 3. Negative Emotion Score by Agent (Boxplot)
        plt.figure(figsize=(12, 7))
        order = df.groupby('agent')['negative_emotion_score'].mean().sort_values(ascending=False).index
        sns.boxplot(x='agent', y='negative_emotion_score', data=df, order=order, palette="Reds")
        plt.title("Negative Emotion Score by Agent", fontsize=14, fontweight='bold')
        plt.xlabel("Agent", fontsize=12)
        plt.ylabel("Negative Emotion Score", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(emotions_dir, "negative_emotion_by_agent.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 4. Individual Emotion Scores Heatmap by Agent
        plt.figure(figsize=(12, 8))
        emotion_cols = ['emotion_anger', 'emotion_disgust', 'emotion_fear',
                        'emotion_joy', 'emotion_sadness', 'emotion_surprise', 'emotion_neutral']
        emotion_means = df.groupby('agent')[emotion_cols].mean()
        emotion_means.columns = [c.replace('emotion_', '') for c in emotion_means.columns]
        sns.heatmap(emotion_means, annot=True, fmt='.3f', cmap='RdYlGn_r', cbar_kws={'label': 'Mean Score'})
        plt.title("Mean Emotion Scores by Agent", fontsize=14, fontweight='bold')
        plt.xlabel("Emotion", fontsize=12)
        plt.ylabel("Agent", fontsize=12)
        plt.tight_layout()
        plot_path = os.path.join(emotions_dir, "emotion_heatmap_by_agent.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        # 5. AI vs Human Emotion Comparison
        plt.figure(figsize=(10, 6))
        df['is_human'] = df['agent'] == 'Human'
        ai_emotions = df[~df['is_human']][emotion_cols].mean()
        human_emotions = df[df['is_human']][emotion_cols].mean()

        x = np.arange(len(emotion_cols))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, ai_emotions, width, label='AI Agents', color='steelblue')
        bars2 = ax.bar(x + width/2, human_emotions, width, label='Human', color='coral')
        ax.set_xlabel('Emotion', fontsize=12)
        ax.set_ylabel('Mean Score', fontsize=12)
        ax.set_title('Emotion Comparison: AI Agents vs Human', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('emotion_', '') for c in emotion_cols], rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plot_path = os.path.join(emotions_dir, "emotion_ai_vs_human.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
        plt.close()

        print(f"   Emotion visualizations saved to: {emotions_dir}")

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

        # =====================================================
        # 7. STATISTICAL TESTS FOR PR TYPE DIFFERENCES
        # =====================================================
        print("\n   Running statistical tests for PR type differences...")

        # Filter to PR types with sufficient samples (n >= 30)
        type_counts = df_typed['pr_type'].value_counts()
        valid_types = type_counts[type_counts >= 30].index.tolist()
        df_valid = df_typed[df_typed['pr_type'].isin(valid_types)]

        print(f"   PR types with n >= 30: {valid_types}")

        if len(valid_types) >= 2:
            # Kruskal-Wallis test
            groups = [df_valid[df_valid['pr_type'] == t]['friction_score'].values for t in valid_types]
            h_stat, kw_pvalue = stats.kruskal(*groups)

            # Effect size (eta-squared)
            n_total = len(df_valid)
            eta_sq = (h_stat - len(valid_types) + 1) / (n_total - len(valid_types))

            print(f"   Kruskal-Wallis: H={h_stat:.2f}, p={kw_pvalue:.2e}, η²={eta_sq:.4f}")

            # Store KW results
            self.results['pr_type_kruskal_wallis'] = {
                'h_statistic': h_stat,
                'p_value': kw_pvalue,
                'eta_squared': eta_sq,
                'n_groups': len(valid_types),
                'n_total': n_total,
                'significant': kw_pvalue < 0.05
            }

            # Dunn's post-hoc test if significant
            if kw_pvalue < 0.05:
                print("   Running Dunn's post-hoc test...")
                import scikit_posthocs as sp

                # Dunn's test with Bonferroni correction
                dunn_results = sp.posthoc_dunn(
                    df_valid, val_col='friction_score', group_col='pr_type', p_adjust='bonferroni'
                )

                # Convert to pairwise format
                pairwise_results = []
                for i, type1 in enumerate(valid_types):
                    for type2 in valid_types[i+1:]:
                        p_bonf = dunn_results.loc[type1, type2]

                        # Get raw p-value (re-run without correction)
                        dunn_raw = sp.posthoc_dunn(
                            df_valid, val_col='friction_score', group_col='pr_type', p_adjust=None
                        )
                        p_raw = dunn_raw.loc[type1, type2]

                        # Holm correction
                        dunn_holm = sp.posthoc_dunn(
                            df_valid, val_col='friction_score', group_col='pr_type', p_adjust='holm'
                        )
                        p_holm = dunn_holm.loc[type1, type2]

                        # Benjamini-Hochberg correction
                        dunn_bh = sp.posthoc_dunn(
                            df_valid, val_col='friction_score', group_col='pr_type', p_adjust='fdr_bh'
                        )
                        p_bh = dunn_bh.loc[type1, type2]

                        # Cliff's Delta effect size
                        data1 = df_valid[df_valid['pr_type'] == type1]['friction_score'].values
                        data2 = df_valid[df_valid['pr_type'] == type2]['friction_score'].values

                        n1, n2 = len(data1), len(data2)
                        more = np.sum(data1[:, None] > data2)
                        less = np.sum(data1[:, None] < data2)
                        cliff_delta = (more - less) / (n1 * n2)

                        # Effect size interpretation
                        abs_delta = abs(cliff_delta)
                        if abs_delta < 0.147:
                            effect_interp = "negligible"
                        elif abs_delta < 0.33:
                            effect_interp = "small"
                        elif abs_delta < 0.474:
                            effect_interp = "medium"
                        else:
                            effect_interp = "large"

                        pairwise_results.append({
                            'pair': f"{type1} vs {type2}",
                            'type1': type1,
                            'type2': type2,
                            'n1': n1,
                            'n2': n2,
                            'p_raw': p_raw,
                            'p_bonferroni': p_bonf,
                            'p_holm': p_holm,
                            'p_bh': p_bh,
                            'cliff_delta': cliff_delta,
                            'effect_size': effect_interp,
                            'significant_bonferroni': p_bonf < 0.05,
                            'significant_holm': p_holm < 0.05,
                            'significant_bh': p_bh < 0.05
                        })

                self.results['pr_type_dunn_test'] = pd.DataFrame(pairwise_results)
                print(f"   ✓ Dunn's test complete: {len(pairwise_results)} pairwise comparisons")

        # =====================================================
        # 8. DETAILED AGENT × PR TYPE CROSS ANALYSIS
        # =====================================================
        print("\n   Computing Agent × PR Type cross analysis...")

        # Exclude Human (insufficient samples) for statistical analysis
        df_cross = df_valid[df_valid['agent'] != 'Human'].copy()

        # Cross-tabulation statistics
        cross_stats = df_cross.groupby(['agent', 'pr_type']).agg({
            'friction_score': ['mean', 'std', 'count'],
            'is_negative': 'mean'
        }).round(4)
        cross_stats.columns = ['mean_friction', 'std_friction', 'count', 'negative_rate']
        cross_stats = cross_stats.reset_index()
        self.results['agent_pr_type_cross'] = cross_stats

        # Two-way analysis: Is there interaction between agent and pr_type?
        # Use 2-way ANOVA approximation with aligned ranks
        print("   Testing Agent × PR Type interaction...")

        # For each PR type, test if agents differ
        pr_type_agent_tests = []
        for pr_type in valid_types:
            df_prtype = df_cross[df_cross['pr_type'] == pr_type]
            agents_in_type = df_prtype['agent'].unique()

            if len(agents_in_type) >= 2:
                agent_groups = [df_prtype[df_prtype['agent'] == a]['friction_score'].values
                               for a in agents_in_type if len(df_prtype[df_prtype['agent'] == a]) >= 10]

                if len(agent_groups) >= 2:
                    h_stat, p_val = stats.kruskal(*agent_groups)
                    pr_type_agent_tests.append({
                        'pr_type': pr_type,
                        'n_agents': len(agent_groups),
                        'n_total': len(df_prtype),
                        'kw_h_statistic': h_stat,
                        'kw_p_value': p_val,
                        'significant': p_val < 0.05
                    })

        self.results['pr_type_agent_kw_tests'] = pd.DataFrame(pr_type_agent_tests)
        print(f"   ✓ Tested agent differences within {len(pr_type_agent_tests)} PR types")

        # For each agent, test if PR types differ
        agent_pr_type_tests = []
        agents = df_cross['agent'].unique()
        for agent in agents:
            df_agent = df_cross[df_cross['agent'] == agent]
            types_for_agent = [t for t in valid_types
                              if len(df_agent[df_agent['pr_type'] == t]) >= 10]

            if len(types_for_agent) >= 2:
                type_groups = [df_agent[df_agent['pr_type'] == t]['friction_score'].values
                              for t in types_for_agent]

                h_stat, p_val = stats.kruskal(*type_groups)
                agent_pr_type_tests.append({
                    'agent': agent,
                    'n_pr_types': len(types_for_agent),
                    'n_total': len(df_agent),
                    'kw_h_statistic': h_stat,
                    'kw_p_value': p_val,
                    'significant': p_val < 0.05
                })

        self.results['agent_pr_type_kw_tests'] = pd.DataFrame(agent_pr_type_tests)
        print(f"   ✓ Tested PR type differences within {len(agent_pr_type_tests)} agents")

        # =====================================================
        # 9. ADDITIONAL VISUALIZATIONS FOR CROSS ANALYSIS
        # =====================================================

        # Grouped bar chart: Friction by Agent for each major PR type
        major_types = ['feat', 'fix', 'docs', 'refactor', 'test', 'chore']
        major_types = [t for t in major_types if t in valid_types]

        if len(major_types) >= 2:
            plt.figure(figsize=(16, 8))
            df_major = df_cross[df_cross['pr_type'].isin(major_types)]

            # Pivot for grouped bar
            pivot_data = df_major.groupby(['pr_type', 'agent'])['friction_score'].mean().unstack()

            pivot_data.plot(kind='bar', figsize=(16, 8), colormap='tab10', edgecolor='black')
            plt.title("Mean Friction Score by PR Type and Agent", fontsize=14, fontweight='bold')
            plt.xlabel("PR Type", fontsize=12)
            plt.ylabel("Mean Friction Score", fontsize=12)
            plt.legend(title="Agent", bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_path = os.path.join(pr_type_dir, "friction_agent_by_pr_type_grouped.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   Saved: {plot_path}")
            plt.close()

            # Negative rate heatmap
            plt.figure(figsize=(14, 8))
            neg_pivot = df_cross.groupby(['pr_type', 'agent'])['is_negative'].mean().unstack()
            sns.heatmap(neg_pivot, annot=True, fmt='.2%', cmap='Reds', cbar_kws={'label': 'Negative Rate'})
            plt.title("Negative Sentiment Rate: PR Type vs Agent", fontsize=14, fontweight='bold')
            plt.xlabel("Agent", fontsize=12)
            plt.ylabel("PR Type", fontsize=12)
            plt.tight_layout()
            plot_path = os.path.join(pr_type_dir, "negative_rate_heatmap_type_agent.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   Saved: {plot_path}")
            plt.close()

        print(f"   PR type visualizations and statistical analysis complete.")

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

        # 3a. Save topics by language (multilingual topic modeling)
        if 'topics_by_lang' in self.results:
            topics_lang_dir = os.path.join(data_dir, "topics_by_language")
            os.makedirs(topics_lang_dir, exist_ok=True)
            for lang, topic_df in self.results['topics_by_lang'].items():
                lang_path = os.path.join(topics_lang_dir, f"topics_{lang}.csv")
                topic_df.to_csv(lang_path, index=False)
            print(f"   ✓ Saved topics by language: {topics_lang_dir}/ ({len(self.results['topics_by_lang'])} languages)")

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

        # 3g. Save pairwise Dunn's test results with multiple corrections
        # Now using Dunn's test (proper post-hoc for KW) instead of Mann-Whitney U
        if 'pairwise_tests' in self.results:
            pairwise_path = os.path.join(data_dir, "pairwise_dunn_test.csv")
            pairwise_df = pd.DataFrame(self.results['pairwise_tests'])
            pairwise_df.to_csv(pairwise_path, index=False)
            print(f"   ✓ Saved Dunn's test results (Bonferroni, Holm, BH): {pairwise_path}")

        # 3h. Save confounding variable analysis results
        if 'confounder_analysis' in self.results:
            conf_analysis = self.results['confounder_analysis']

            # Save regression coefficients
            if conf_analysis.get('model_adjusted', {}).get('coefficients'):
                coef_data = []
                for var, vals in conf_analysis['model_adjusted']['coefficients'].items():
                    coef_data.append({
                        'variable': var,
                        'coefficient': vals['coef'],
                        'p_value': vals['pval'],
                        'ci_low': vals.get('ci_low', None),
                        'ci_high': vals.get('ci_high', None)
                    })
                coef_df = pd.DataFrame(coef_data)
                coef_path = os.path.join(data_dir, "confounder_regression_coefficients.csv")
                coef_df.to_csv(coef_path, index=False)
                print(f"   ✓ Saved confounder analysis: {coef_path}")

            # Save summary
            conf_summary = {
                'r_squared_unadjusted': conf_analysis.get('model_unadjusted', {}).get('rsquared'),
                'r_squared_adjusted': conf_analysis.get('model_adjusted', {}).get('rsquared'),
                'r_squared_adj_adjusted': conf_analysis.get('model_adjusted', {}).get('rsquared_adj'),
                'n_confounders': conf_analysis.get('model_adjusted', {}).get('n_confounders'),
                'reference_agent': conf_analysis.get('reference_agent'),
                'n_observations': conf_analysis.get('n_observations')
            }
            conf_summary_df = pd.DataFrame([conf_summary])
            conf_summary_path = os.path.join(data_dir, "confounder_analysis_summary.csv")
            conf_summary_df.to_csv(conf_summary_path, index=False)

        # 3i. Save power analysis results
        if 'power_analysis_omnibus' in self.results:
            power_omnibus = self.results['power_analysis_omnibus']
            power_omnibus_df = pd.DataFrame([power_omnibus])
            power_omnibus_path = os.path.join(data_dir, "power_analysis_omnibus.csv")
            power_omnibus_df.to_csv(power_omnibus_path, index=False)
            print(f"   ✓ Saved omnibus power analysis: {power_omnibus_path}")

        if 'power_analysis_pairwise' in self.results:
            power_pairwise_df = pd.DataFrame(self.results['power_analysis_pairwise'])
            power_pairwise_path = os.path.join(data_dir, "power_analysis_pairwise.csv")
            power_pairwise_df.to_csv(power_pairwise_path, index=False)
            print(f"   ✓ Saved pairwise power analysis: {power_pairwise_path}")

        if 'power_sensitivity' in self.results:
            power_sens_df = pd.DataFrame([self.results['power_sensitivity']])
            power_sens_path = os.path.join(data_dir, "power_sensitivity_analysis.csv")
            power_sens_df.to_csv(power_sens_path, index=False)
            print(f"   ✓ Saved power sensitivity analysis: {power_sens_path}")

        # 3j. Save multi-model sentiment analysis results
        if 'multimodel_sentiment' in self.results:
            mm_results = self.results['multimodel_sentiment']

            # Save pairwise agreement (Cohen's Kappa)
            if mm_results.get('pairwise_agreement'):
                agreement_df = pd.DataFrame(mm_results['pairwise_agreement'])
                agreement_path = os.path.join(data_dir, "multimodel_intermodel_agreement.csv")
                agreement_df.to_csv(agreement_path, index=False)
                print(f"   ✓ Saved inter-model agreement (Cohen's κ): {agreement_path}")

            # Save model comparison summary
            model_summary = []
            for model_name, model_stats in mm_results.get('model_predictions', {}).items():
                model_summary.append({
                    'model': model_name,
                    'n_negative': model_stats.get('n_negative'),
                    'n_positive': model_stats.get('n_positive'),
                    'n_neutral': model_stats.get('n_neutral'),
                    'mean_neg_score': model_stats.get('mean_neg_score')
                })
            if model_summary:
                model_summary_df = pd.DataFrame(model_summary)
                model_summary_path = os.path.join(data_dir, "multimodel_comparison_summary.csv")
                model_summary_df.to_csv(model_summary_path, index=False)
                print(f"   ✓ Saved multi-model comparison: {model_summary_path}")

            # Save ensemble statistics
            if mm_results.get('ensemble'):
                ensemble_summary = {
                    'n_samples': mm_results.get('n_samples'),
                    'models_used': ', '.join(mm_results.get('models_used', [])),
                    **mm_results['ensemble']
                }
                ensemble_df = pd.DataFrame([ensemble_summary])
                ensemble_path = os.path.join(data_dir, "multimodel_ensemble_summary.csv")
                ensemble_df.to_csv(ensemble_path, index=False)
                print(f"   ✓ Saved ensemble summary: {ensemble_path}")

        # 3k. Save emotion statistics by agent
        if 'dominant_emotion' in self.analyzed_df.columns:
            emotion_stats_path = os.path.join(data_dir, "emotion_stats_by_agent.csv")
            emotion_cols = ['emotion_anger', 'emotion_disgust', 'emotion_fear',
                           'emotion_joy', 'emotion_sadness', 'emotion_surprise', 'emotion_neutral',
                           'negative_emotion_score']
            available_cols = [c for c in emotion_cols if c in self.analyzed_df.columns]
            if available_cols:
                emotion_stats = self.analyzed_df.groupby('agent')[available_cols].mean().reset_index()
                emotion_stats.to_csv(emotion_stats_path, index=False)
                print(f"   ✓ Saved emotion stats: {emotion_stats_path}")

        # 3l. Save PR Type statistics and analysis
        if 'pr_type_stats' in self.results:
            pr_type_stats_path = os.path.join(data_dir, "pr_type_friction_stats.csv")
            self.results['pr_type_stats'].to_csv(pr_type_stats_path, index=False)
            print(f"   ✓ Saved PR type friction stats: {pr_type_stats_path}")

        if 'pr_type_kruskal_wallis' in self.results:
            pr_type_kw = self.results['pr_type_kruskal_wallis']
            pr_type_kw_df = pd.DataFrame([pr_type_kw])
            pr_type_kw_path = os.path.join(data_dir, "pr_type_kruskal_wallis.csv")
            pr_type_kw_df.to_csv(pr_type_kw_path, index=False)
            print(f"   ✓ Saved PR type Kruskal-Wallis test: {pr_type_kw_path}")

        if 'pr_type_dunn_test' in self.results:
            pr_type_dunn_path = os.path.join(data_dir, "pr_type_dunn_test.csv")
            self.results['pr_type_dunn_test'].to_csv(pr_type_dunn_path, index=False)
            print(f"   ✓ Saved PR type Dunn's post-hoc test: {pr_type_dunn_path}")

        if 'agent_pr_type_cross' in self.results:
            cross_path = os.path.join(data_dir, "agent_pr_type_cross_stats.csv")
            self.results['agent_pr_type_cross'].to_csv(cross_path, index=False)
            print(f"   ✓ Saved Agent × PR Type cross stats: {cross_path}")

        if 'pr_type_agent_kw_tests' in self.results:
            pr_agent_kw_path = os.path.join(data_dir, "pr_type_agent_kw_tests.csv")
            self.results['pr_type_agent_kw_tests'].to_csv(pr_agent_kw_path, index=False)
            print(f"   ✓ Saved agent differences within PR types: {pr_agent_kw_path}")

        if 'agent_pr_type_kw_tests' in self.results:
            agent_pr_kw_path = os.path.join(data_dir, "agent_pr_type_kw_tests.csv")
            self.results['agent_pr_type_kw_tests'].to_csv(agent_pr_kw_path, index=False)
            print(f"   ✓ Saved PR type differences within agents: {agent_pr_kw_path}")

        # 4. Save statistical test results (with effect sizes)
        stats_summary = {
            "kruskal_wallis_agents_stat": self.results.get('kruskal_wallis_agents', {}).get('stat', None),
            "kruskal_wallis_agents_pvalue": self.results.get('kruskal_wallis_agents', {}).get('p_value', None),
            "kruskal_wallis_eta_squared": self.results.get('kruskal_wallis_agents', {}).get('eta_squared', None),
            "kruskal_wallis_effect_size": self.results.get('kruskal_wallis_agents', {}).get('effect_size', None),
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

        # 5. Save topic models (pickle) - per language
        if hasattr(self, 'topic_models_by_lang') and self.topic_models_by_lang:
            topic_models_dir = os.path.join(models_dir, "bertopic_by_language")
            os.makedirs(topic_models_dir, exist_ok=True)
            for lang, model in self.topic_models_by_lang.items():
                model_path = os.path.join(topic_models_dir, f"bertopic_{lang}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            print(f"   ✓ Saved BERTopic models: {topic_models_dir}/ ({len(self.topic_models_by_lang)} languages)")
        elif hasattr(self, 'topic_model'):
            # Backward compatibility: save single model if no per-language models
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

            # Methodological notes
            f.write("\n" + "-" * 70 + "\n")
            f.write("METHODOLOGICAL NOTES:\n")
            f.write("-" * 70 + "\n\n")

            f.write("1. Human Baseline Exclusion:\n")
            f.write("   The AIDev dataset does not include review comments for Human PRs.\n")
            f.write("   Human data is excluded from statistical tests due to insufficient\n")
            f.write("   sample size (< 100 samples). Human data is retained for descriptive\n")
            f.write("   statistics only.\n\n")

            f.write("2. Category Friction Scores:\n")
            f.write("   Category friction statistics (Testing, Security, Code Style, etc.)\n")
            f.write("   are calculated ONLY on negative comments. This is by design:\n")
            f.write("   - Friction categories are assigned only to negative comments\n")
            f.write("   - Mean friction scores for categories (0.6-0.7) are higher than\n")
            f.write("     per-agent means (0.1-0.3) because they exclude neutral/positive\n")
            f.write("   - Per-agent means include all comments regardless of sentiment\n\n")

            f.write("3. Multi-Model Validation:\n")
            f.write("   All sentiment models use 3-class output (negative/neutral/positive)\n")
            f.write("   to ensure valid Cohen's Kappa inter-model agreement calculation.\n")

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

        # Phase 2: Sentiment analysis (primary model)
        self.analyze_sentiment()

        # Phase 2a: Multi-model sentiment analysis (robustness check)
        self.analyze_sentiment_multimodel()

        # Phase 2b: Emotion analysis (Ekman 7 categories)
        self.analyze_emotions()

        # Phase 3: Topic modeling
        self.extract_friction_topics()

        # Phase 3b: Category classification (Enhanced RQ2)
        self.classify_friction_categories()

        # Phase 4: Statistical analysis (RQ1, RQ3, RQ4)
        self.analyze_outcomes()

        # Phase 4b: Category-based friction analysis
        self.analyze_category_friction()

        # Phase 4c: Confounding variable analysis (controls for PR type, source, duration)
        self.analyze_confounders()

        # Phase 4d: Statistical power analysis
        self.analyze_power()

        # Enhanced Research Questions
        self.analyze_temporal_evolution()  # RQ5
        self.analyze_topic_agent_interaction()  # Enhanced RQ2
        self.analyze_timemerge_iterations()  # Enhanced RQ4

        # Phase 5 & 6: Visualization and saving
        self.visualize_results()
        self.visualize_emotions()  # Emotion visualizations (Ekman 7 categories)
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
