from literature_recommender.scraper import scrape_article


def test_scraper_basic_functionality():
    test_url = 'https://en.wikipedia.org/wiki/Franz_Kafka'
    print(f'1. Testing scraper on: {test_url}')
    art = scrape_article(test_url)
    assert art is not None, 'Scraper returned None'
    assert len(art['text']) > 300, 'Scraped text too short'
    assert len(art['summary']) > 0, 'Summary missing'