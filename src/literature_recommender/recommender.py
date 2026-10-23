import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from literature_recommender.config import (
    DEFAULT_HYBRID_ALPHA,
    DENSE_EMBEDDINGS_FILE,
    PREPROCESSED_PARQUET,
    SENTENCE_TRANSFORMER_MODEL,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_MODEL_FILE,
    WEIGHTS_DIR,
)
from literature_recommender.scraper import scrape_article
from literature_recommender.preprocessor import preprocess_text


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

    def get_article_vectors(self, link, debug=False):
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
                article_vectors = self.get_article_vectors(link, debug=debug)
                dense_vectors.append(article_vectors["dense_vector"])
                tfidf_vectors.append(article_vectors["tfidf_vector"])

                if article_vectors["in_database"]:
                    excluded_indices.append(article_vectors["db_index"])

            except Exception as e:
                if debug:
                    print(f"Skipping link: {link}. Due to error: {e}")

        if not dense_vectors or not tfidf_vectors:
            raise ValueError("None of the links could be successfully processed.")

        user_dense_profile = np.mean(dense_vectors, axis=0)
        user_dense_profile_norm = np.linalg.norm(user_dense_profile)
        if user_dense_profile_norm > 0:
            user_dense_profile = user_dense_profile / user_dense_profile_norm

        user_tfidf_profile = np.mean(tfidf_vectors, axis=0, keepdims=True)

        dense_similarities = np.dot(self.dense_matrix, user_dense_profile)
        tfidf_similarities = cosine_similarity(self.tfidf_matrix, user_tfidf_profile).flatten()

        match method:
            case 'dense':
                final_similarities = dense_similarities
            case 'tfidf':
                final_similarities = tfidf_similarities
            case 'hybrid':
                final_similarities = (alpha * min_max_norm(dense_similarities)) + ((1-alpha) * min_max_norm(tfidf_similarities))
            case _:
                raise ValueError(f"Invalid method: {method}. Choose 'dense', 'tfidf', 'hybrid'.")

        results_df = self.df.copy()
        results_df['final_score'] = final_similarities
        results_df['dense_score'] = dense_similarities
        results_df['tfidf_score'] = tfidf_similarities

        if excluded_indices:
            results_df = results_df.drop(index=excluded_indices)

        top_recommendations = results_df.nlargest(top_k, 'final_score')

        return top_recommendations[['title', 'url', 'final_score', 'dense_score', 'tfidf_score', 'summary']]




