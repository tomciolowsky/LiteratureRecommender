from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
SCRAPING_DATA_DIR = DATA_DIR / "scraping"
RAW_DATA_FILE = SCRAPING_DATA_DIR / "scraped_data.jsonl"
ADDITIONAL_LINKS_FILE = SCRAPING_DATA_DIR / "additional_links.txt"
PREPROCESSED_PARQUET = DATA_DIR / "wiki_literature_preprocessed.parquet"
PREPROCESSED_CSV = DATA_DIR / "wiki_literature_preprocessed.csv"

WEIGHTS_DIR = PROJECT_ROOT / "weights"
DENSE_EMBEDDINGS_FILE = WEIGHTS_DIR / "dense_embeddings.npy"
TFIDF_MODEL_FILE = WEIGHTS_DIR / "tfidf_model.joblib"

PLOTS_DIR = PROJECT_ROOT / "plots"

HEADERS = {
    "User-Agent": "LiteratureRecommender/1.0 (contact: tomasz.bialecki@student.put.poznan.pl)"
}
TARGET_ARTICLE_COUNT = 1200
REQUEST_DELAY_MULTIPLIER = 3
STARTING_URLS = {
    "Le Monde": "https://en.wikipedia.org/wiki/Le_Monde%27s_100_Books_of_the_Century",
    "Nobel Prize": "https://en.wikipedia.org/wiki/List_of_Nobel_laureates_in_Literature",
    "Pulitzer Prize": "https://en.wikipedia.org/wiki/Pulitzer_Prize_for_Fiction",
}

DOMAIN_STOPWORDS = {
    "isbn",
    "oclc",
    "doi",
    "issn",
    "page",
    "pages",
    "volume",
    "vol",
    "retrieved",
    "archived",
    "original",
    "cite",
    "citation",
    "press",
    "edition",
    "editions",
    "publisher",
    "published",
    "publication",
    "chapter",
    "author",
    "literary",
    "literature",
    "also",
    "often",
    "would",
    "one",
    "first",
    "two",
    "new",
    "well",
    "known",
    "may",
    "many",
    "see",
}

SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.8
DEFAULT_HYBRID_ALPHA = 0.70