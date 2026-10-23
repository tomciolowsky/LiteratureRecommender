from literature_recommender.preprocessor import clean_raw_text, preprocess_text


def test_clean_raw_text_removes_urls_and_special_chars():
    raw = "Visit https://en.wikipedia.org/wiki/Franz_Kafka for Kafka's 1915 novel!"
    cleaned = clean_raw_text(raw)
    assert "https" not in cleaned
    assert "1915" not in cleaned
    assert "kafka" in cleaned


def test_preprocess_text_lemmatization_and_stopwords():
    text = "The author published several philosophical novels about existentialism."
    res = preprocess_text(text)

    # Domain stopwords check ('author', 'published')
    assert "author" not in res["tokens_lemmatized"]
    assert "published" not in res["tokens_lemmatized"]

    # Lemmatization check ('novels' -> 'novel')
    assert "novel" in res["tokens_lemmatized"]
    assert res["token_count"] > 0
    assert res["vocab_size"] == len(set(res["tokens_lemmatized"]))