# LiteratureRecommender

![CI Pipeline](https://github.com/tomciolowsky/LiteratureRecommender/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.14-blue.svg?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9.svg?logo=uv)

An Information Retrieval and Recommendation System that accepts a collection of Wikipedia links (both cached in the database and unseen) and surfaces related classic books and authors with visual prediction breakdowns.

## Highlights

- **Scraping without Topic Drift:** First collecting links from Le Monde's 100 Books of the Century, Nobel Laureates, and Pulitzer Winners. Then extracting 'Notable Works' from authors. Finally scraping all gathered links.
- **Hybrid Scoring Engine:** Blending dense embeddings vectors from (`all-MiniLM-L6-v2`) with sparse lexical matching (TF-IDF over WordNet-lemmatized texts).
- **Dynamic Inference for Unseen URLs:** Paste **any** Wikipedia article link; the engine scrapes, tokenizes, and projects it into the precomputed vector space on the fly.
- **Visual Interpretability:** Word-level contribution attribution charts, score decomposition bars, and multi-source profile influence plots.
- **Self-Expanding Database:** Newly scraped unseen articles can be merged and re-indexed into the Parquet corpus at the click of a button.

## Mathematical Formulation

### 1. User Profile Centroid
For a user collection of $K$ visited articles $\mathcal{A}_{\text{visited}} = \{a_1, a_2, \dots, a_K\}$:

$$\vec{u}_{\text{dense}} = \frac{1}{K} \sum_{i=1}^K \vec{v}_{a_i}, \quad \hat{u}_{\text{dense}} = \frac{\vec{u}_{\text{dense}}}{\Vert{}\vec{u}_{\text{dense}}\Vert{}_2}$$

$$\vec{u}_{\text{tfidf}} = \frac{1}{K} \sum_{i=1}^K \vec{w}_{a_i}$$

### 2. Hybrid Scoring
Dense similarities ($S_{\text{dense}} = \hat{u}_{\text{dense}} \cdot \hat{v}_d$) and sparse cosine similarities ($S_{\text{tfidf}}$) are Min-Max calibrated before linear combination:

$$\tilde{S}(d) = \frac{S(d) - \min(S)}{\max(S) - \min(S)}$$

$$S_{\text{final}}(d) = \alpha \cdot \tilde{S}_{\text{dense}}(d) + (1 - \alpha) \cdot \tilde{S}_{\text{tfidf}}(d) \quad (\alpha = 0.70)$$

### 3. Word Attribution Breakdown
$$\text{Contribution}(w) = \vec{w}_{\text{query}}[w] \times \vec{w}_{\text{rec}}[w]$$


## Installation

### Using uv:

Requires [uv](https://github.com/astral-sh/uv) package manager

```bash
git clone https://github.com/tomciolowsky/LiteratureRecommender.git
cd LiteratureRecommender 
uv sync
```

### Using pip

Standard Installation:

```bash
git clone https://github.com/tomciolowsky/LiteratureRecommender.git
cd LiteratureRecommender
pip install .
```

Development Installation (modify or run tests):

```bash
git clone https://github.com/tomciolowsky/LiteratureRecommender.git
cd LiteratureRecommender
pip install -e .[dev]
```

## Interactive Web Application (Streamlit)

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) to explore recommendations, examine breakdown plots, and sync new articles.

## Command Line Interface (CLI)

Examples of how to use:

```bash
# Single article recommendation
uv run recommend https://en.wikipedia.org/wiki/Franz_Kafka

# Multi-article composite profile with custom top-k
uv run recommend https://en.wikipedia.org/wiki/Jean-Paul_Sartre https://en.wikipedia.org/wiki/Albert_Camus --top-k 3
```

CLI detailed instructions:

```text
positional arguments:
  urls                              One or more Wikipedia URLs (e.g., https://en.wikipedia.org/wiki/Franz_Kafka)

options:
  -h, --help                        show this help message and exit
  -k, --top-k TOP_K                 Number of recommendations (default: 5)
  -m, --method {hybrid,dense,tfidf} Retrieval method
                        
  -a, --alpha ALPHA                 Weight for dense similarity in hybrid mode
  -v, --verbose                     Display additional information
```

## Testing and Code linting

Use `pytest` for unit testing:

```bash
# Run tests
uv run pytest -v
```

Use `ruff` for code linting and quality checks:
```bash
# Run linter
uv run ruff check .
```

## Repository Structure

```text
LiteratureRecommender/
├── .github/workflows/ci.yml       # Automated GitHub Actions test pipeline
├── data/                          # Processed Parquet & CSV databases
├── notebooks/                     # Step-by-step development & report.ipynb
├── plots/                         # Static report figures
├── src/literature_recommender/    # Core package
│   ├── config.py                  # Paths and hyperparameter settings
│   ├── scraper.py                 # Wikipedia parsing & cleaning
│   ├── preprocessor.py            # Lemmatization & token filtering
│   ├── recommender.py             # Hybrid engine & database re-indexer
│   ├── visualizer.py              # Matplotlib visual breakdowns
│   └── cli.py                     # Command-line entrypoint
├── tests/                         # Unit test suite
├── weights/                       # Serialized TF-IDF and dense embeddings
├── app.py                         # Streamlit dashboard
├── pyproject.toml                 # Package specs & dependencies
└── uv.lock                        # Deterministic dependency lockfile
```