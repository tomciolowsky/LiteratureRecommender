import numpy as np
import pytest

from literature_recommender.recommender import LiteratureRecommender, min_max_norm


def test_min_max_norm_scaling():
    arr = np.array([2.0, 4.0, 6.0, 10.0])
    norm = min_max_norm(arr)
    assert norm.min() == pytest.approx(0.0)
    assert norm.max() == pytest.approx(1.0)
    assert norm[1] == pytest.approx(0.25)


def test_min_max_norm_constant_array():
    arr = np.array([5.0, 5.0, 5.0])
    norm = min_max_norm(arr)
    np.testing.assert_array_equal(norm, arr)


@pytest.fixture(scope="module")
def recommender():
    return LiteratureRecommender(load_precomputed=True)


def test_recommender_retrieval_output_schema(recommender):
    seed_url = "https://en.wikipedia.org/wiki/Franz_Kafka"
    top_k = 3
    recs = recommender.recommend_articles([seed_url], top_k=top_k, method="hybrid")

    assert len(recs) == top_k
    assert {"title", "url", "final_score", "dense_score", "tfidf_score", "summary"}.issubset(
        recs.columns
    )
    assert seed_url.lower() not in recs["url"].str.lower().values
    assert recs["final_score"].is_monotonic_decreasing