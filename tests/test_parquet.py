import os

import pandas as pd


def test_parquet_write_read():
    art = {
        'title': 'Franz Kafka',
        'url': 'https://en.wikipedia.org/wiki/Franz_Kafka',
        'summary': 'Franz Kafka was a German-speaking Bohemian novelist and short-story writer.',
        'categories': ['1883 births', '1924 deaths']
    }
    proc = {
        'text_lemmatized': 'Franz Kafka was a German-speaking Bohemian novelist and short-story writer.',
        'text_stemmed': 'Franz Kafka was a German-speaking Bohemian novelist and short-story writer.',
        'token_count': 10,
        'vocab_size': 8
    }
    test_row = {
        'title': art['title'],
        'url': art['url'],
        'summary': art['summary'],
        'categories': art['categories'],
        'text_lemmatized': proc['text_lemmatized'],
        'text_stemmed': proc['text_stemmed'],
        'token_count': proc['token_count'],
        'vocab_size': proc['vocab_size']
    }
    test_df = pd.DataFrame([test_row])
    test_df.to_parquet('test_sample.parquet', index=False)
    read_df = pd.read_parquet('test_sample.parquet')
    assert len(read_df) == 1, 'Parquet write/read mismatch'
    os.remove('test_sample.parquet')