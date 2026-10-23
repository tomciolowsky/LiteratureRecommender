import json
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from literature_recommender.config import (
    DATA_DIR,
    DOMAIN_STOPWORDS,
    PREPROCESSED_CSV,
    PREPROCESSED_PARQUET,
    RAW_DATA_FILE,
)

for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words.update(DOMAIN_STOPWORDS)


def clean_raw_text(text: str) -> str:
    """
    Lowercases, removes URLs, numbers, special characters and extra spaces.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text) # Remove URLs
    text = re.sub(r"[^a-z\s-]", " ", text) # Remove numbers & special characters
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip() # Remove extra whitespace
    return text


def preprocess_text(text: str) -> dict:
    """
    Performs tokenization, stopword removal, stemming, and lemmatization.
    """
    cleaned = clean_raw_text(text)
    tokens = word_tokenize(cleaned)
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) >= 3]

    lemmatized = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    stemmed = [stemmer.stem(w) for w in filtered_tokens]

    return {
        "tokens_clean": filtered_tokens,
        "tokens_lemmatized": lemmatized,
        "tokens_stemmed": stemmed,
        "text_lemmatized": " ".join(lemmatized),
        "text_stemmed": " ".join(stemmed),
        "vocab_size": len(set(lemmatized)),
        "token_count": len(filtered_tokens)
    }


def read_from_file() -> pd.DataFrame:
    """
    Reads the raw JSONL data file and returns a DataFrame.
    """
    records = []
    with open(RAW_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    return df


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies preprocessing to the DataFrame and filters out documents with fewer than 30 tokens.
    """
    tqdm.pandas(desc="Preprocessing Documents")
    processed_results = df["text"].progress_apply(preprocess_text)

    processed_df = pd.DataFrame(list(processed_results))
    df = pd.concat([df, processed_df], axis=1)
    
    df = df[df["token_count"] >= 30].reset_index(drop=True)
    
    return df


def save_to_files(df: pd.DataFrame) -> None:
    """
    Saves the preprocessed DataFrame to Parquet and CSV files.
    """
    save_columns = [
        "title",
        "url",
        "summary",
        "categories",
        "text_lemmatized",
        "text_stemmed",
        "token_count",
        "vocab_size",
    ]

    df[save_columns].to_parquet(PREPROCESSED_PARQUET, index=False)

    csv_df = df[save_columns].copy()
    csv_df["categories"] = csv_df["categories"].apply(lambda cats: "|".join(cats))
    csv_df.to_csv(PREPROCESSED_CSV, index=False, encoding="utf-8")


def preprocess_main(debug=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = read_from_file()
    if debug:
        print(f"Loaded {len(df)} records from {RAW_DATA_FILE}")

    df = process_df(df)
    if debug:
        print(f"Number of documents after preprocessing: {len(df)}")

    save_to_files(df)
    if debug:
        print(f"Preprocessed data saved to {PREPROCESSED_PARQUET} and {PREPROCESSED_CSV}")