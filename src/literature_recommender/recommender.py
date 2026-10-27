import contextlib
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from literature_recommender.config import (
    DEFAULT_HYBRID_ALPHA,
    DENSE_EMBEDDINGS_FILE,
    PENDING_ARTICLES_FILE,
    PREPROCESSED_CSV,
    PREPROCESSED_PARQUET,
    SENTENCE_TRANSFORMER_MODEL,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_MODEL_FILE,
    WEIGHTS_DIR,
)
from literature_recommender.preprocessor import preprocess_text
from literature_recommender.scraper import scrape_article

logger = logging.getLogger(__name__)

def min_max_norm(arr: np.ndarray) -> np.ndarray:
    """
    Normalizes array values to [0, 1] using min-max scaling.
    """
    if (arr.max()-arr.min()) > 0:
        return (arr-arr.min()) / (arr.max()-arr.min())
    else:
        return arr
    

class LiteratureRecommender:
    """
    Recommender engine supporting dense, sparse, and hybrid similarity matching.
    """
    def __init__(self, load_precomputed: bool = True):
        self.df = pd.read_parquet(PREPROCESSED_PARQUET).reset_index(drop=True)
        self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        
        if load_precomputed and DENSE_EMBEDDINGS_FILE.exists() and TFIDF_MODEL_FILE.exists():
            self.dense_matrix = np.load(DENSE_EMBEDDINGS_FILE)
            self.tfidf = joblib.load(TFIDF_MODEL_FILE)
            self.tfidf_matrix = self.tfidf.transform(self.df["text_lemmatized"])

        else:
            self.fit_and_save_models()

    def fit_and_save_models(self) -> None:
        """
        Computes and saves TF-IDF model and dense embedding weights
        """
        self.tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
        )
        self.tfidf_matrix = self.tfidf.fit_transform(
            self.df["text_lemmatized"]
        )
        joblib.dump(self.tfidf, TFIDF_MODEL_FILE)

        self.dense_matrix = self.model.encode(
            self.df["summary"].tolist(),
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,
        )
        np.save(DENSE_EMBEDDINGS_FILE, self.dense_matrix)

    def record_pending_article(self, article_data: dict, processed: dict) -> None:
        """
        Appends unseen article to pending queue if not already queued.
        """
        pending_urls = set()
        if PENDING_ARTICLES_FILE.exists():
            with open(PENDING_ARTICLES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            pending_urls.add(json.loads(line)["url"].lower())
                        except json.JSONDecodeError:
                            continue

        if article_data["url"].lower() in pending_urls:
            return

        record = {
            "title": article_data["title"],
            "url": article_data["url"],
            "summary": article_data["summary"],
            "categories": article_data["categories"],
            "text_lemmatized": processed["text_lemmatized"],
            "text_stemmed": processed["text_stemmed"],
            "token_count": processed["token_count"],
            "vocab_size": processed["vocab_size"],
        }

        with open(PENDING_ARTICLES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def get_article_vectors(self, link: str, debug: bool=False, record_pending: bool=True):
        """
        Retrieves vectors for a given article link from DB or computes them if the article is new.
        """
        link_clean = link.strip()

        matching_rows = self.df[self.df['url'].str.lower() == link_clean.lower()]
        
        if matching_rows.empty:
            if debug:
                print(f"New article - scraping and processing: {link_clean}")
            article_data = scrape_article(link_clean)
            if not article_data:
                raise ValueError(f"Failed to scrape article at: {link_clean}")

            processed = preprocess_text(article_data['text'])

            tfidf_vector = self.tfidf.transform([processed["text_lemmatized"]]).toarray().flatten()
            
            dense_vector = self.model.encode(
                [article_data["summary"]],
                normalize_embeddings=True
            )[0]

            if record_pending:
                self.record_pending_article(article_data, processed)

            return {
                "title": article_data["title"],
                "url": link_clean,
                "dense_vector": dense_vector,
                "tfidf_vector": tfidf_vector,
                "in_database": False,
                "db_index": None
            }

        else:
            if debug:
                print(f"Article found in database: {link_clean}")
            
            index = matching_rows.index[0]

            return {
                "title": self.df.loc[index, 'title'],
                "url": self.df.loc[index, 'url'],
                "dense_vector": self.dense_matrix[index],
                "tfidf_vector": self.tfidf_matrix[index].toarray().flatten(),
                "in_database": True,
                "db_index": index
            }

    def recommend_articles(self,
                           visited_links: list[str], 
                           top_k: int = 5, 
                           method: str = "hybrid", 
                           alpha: float = DEFAULT_HYBRID_ALPHA,
                           debug: bool = False
                           ) -> pd.DataFrame:
        """
        Calculates user profile based on visited articles and returns ranked recommendations based on the specified similarity method.
        """
        dense_vectors = []
        tfidf_vectors = []
        excluded_indices = []

        for link in visited_links:
            try:
                article_vectors = self.get_article_vectors(link, debug=debug, record_pending=True)
                dense_vectors.append(article_vectors["dense_vector"])
                tfidf_vectors.append(article_vectors["tfidf_vector"])

                if article_vectors["in_database"]:
                    excluded_indices.append(article_vectors["db_index"])

            except (ValueError, KeyError, ConnectionError) as e:
                logger.warning("Skipping link '%s' due to error: %s", link, e)
            except Exception:
                logger.exception("Unexpected error processing link '%s'", link)

        if not dense_vectors or not tfidf_vectors:
            raise ValueError("None of the links could be successfully processed.")

        user_dense_profile = np.mean(dense_vectors, axis=0)
        user_dense_profile_norm = np.linalg.norm(user_dense_profile)
        if user_dense_profile_norm > 0:
            user_dense_profile = user_dense_profile / user_dense_profile_norm

        user_tfidf_profile = np.mean(tfidf_vectors, axis=0, keepdims=True)

        dense_similarities = np.dot(self.dense_matrix, user_dense_profile)
        tfidf_similarities = cosine_similarity(self.tfidf_matrix, user_tfidf_profile).flatten()

        results_df = self.df.copy()
        results_df['dense_score'] = dense_similarities
        results_df['tfidf_score'] = tfidf_similarities

        if excluded_indices:
            results_df = results_df.drop(index=excluded_indices)

        cand_dense = results_df["dense_score"].to_numpy()
        cand_tfidf = results_df["tfidf_score"].to_numpy()

        match method:
            case 'dense':
                results_df['final_score'] = cand_dense
            case 'tfidf':
                results_df['final_score'] = cand_tfidf
            case 'hybrid':
                results_df['final_score'] = (alpha * min_max_norm(cand_dense)) + ((1-alpha) * min_max_norm(cand_tfidf))
            case _:
                raise ValueError(f"Invalid method: {method}. Choose 'dense', 'tfidf', 'hybrid'.")

        top_recommendations = results_df.nlargest(top_k, 'final_score')

        return top_recommendations[['title', 'url', 'final_score', 'dense_score', 'tfidf_score', 'summary']]

    def get_pending_count(self) -> int:
        """
        Returns number of unseen articles awaiting database integration.
        """
        if not PENDING_ARTICLES_FILE.exists():
            return 0
        with open(PENDING_ARTICLES_FILE, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def _clear_pending_file(self) -> None:
        """
        Safely truncates and deletes pending queue on Windows without file-lock errors.
        """
        if PENDING_ARTICLES_FILE.exists():
            with contextlib.suppress(OSError):
                with open(PENDING_ARTICLES_FILE, "w", encoding="utf-8"):
                    pass
                PENDING_ARTICLES_FILE.unlink(missing_ok=True)

    def reindex_database(self) -> int:
        """
        Folds all pending articles into the parquet DB, updates TF-IDF,
        stacks new dense embeddings, and saves weights.
        """
        if not PENDING_ARTICLES_FILE.exists():
            return 0

        pending_records = []
        with open(PENDING_ARTICLES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pending_records.append(json.loads(line))

        if not pending_records:
            return 0

        new_df = pd.DataFrame(pending_records)
        new_df = new_df[~new_df["url"].str.lower().isin(self.df["url"].str.lower())]

        if new_df.empty:
            PENDING_ARTICLES_FILE.unlink(missing_ok=True)
            return 0
        
        def update_df():
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self.df.to_parquet(PREPROCESSED_PARQUET, index=False)
            csv_df = self.df.copy()
            csv_df["categories"] = csv_df["categories"].apply(
                lambda cats: "|".join(cats)
            )
            csv_df.to_csv(PREPROCESSED_CSV, index=False, encoding="utf-8")

        def update_dense():
            new_dense_vectors = self.model.encode(
                new_df["summary"].tolist(),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            self.dense_matrix = np.vstack([self.dense_matrix, new_dense_vectors])
            np.save(DENSE_EMBEDDINGS_FILE, self.dense_matrix)

        def update_tfidf():
            self.tfidf = TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                min_df=TFIDF_MIN_DF,
                max_df=TFIDF_MAX_DF,
            )
            self.tfidf_matrix = self.tfidf.fit_transform(self.df["text_lemmatized"])
            joblib.dump(self.tfidf, TFIDF_MODEL_FILE)

        update_df()
        update_dense()
        update_tfidf()

        self._clear_pending_file()
        return len(new_df)