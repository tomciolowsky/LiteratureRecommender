import argparse
import os
import warnings
from literature_recommender.recommender import LiteratureRecommender

try:
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
except ImportError:
    pass

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

def main():
    parser = argparse.ArgumentParser(
        description="Wikipedia Classic Literature Recommender"
    )
    parser.add_argument(
        "urls",
        nargs="+",
        help="One or more Wikipedia URLs (e.g., https://en.wikipedia.org/wiki/Franz_Kafka)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of recommendations (default: 5)",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["hybrid", "dense", "tfidf"],
        default="hybrid",
        help="Retrieval method",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.70,
        help="Weight for dense similarity in hybrid mode",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display additional information",
    )

    args = parser.parse_args()
    debug = args.verbose

    if debug:
        print("\nLoading models and indexed literature database...")
    
    recommender = LiteratureRecommender()

    if debug:
        print(f"Generating recommendations for {len(args.urls)} input URL(s)...")
    
    recs = recommender.recommend_articles(
        visited_links=args.urls,
        top_k=args.top_k,
        method=args.method,
        alpha=args.alpha,
        debug=debug,
    )

    print("\n" + "=" * 80)
    print(f"TOP {args.top_k} RECOMMENDATIONS ({args.method.upper()} METHOD)")
    print("=" * 80)
    
    for idx, row in recs.reset_index(drop=True).iterrows():
        print(f"[{idx + 1}] {row['title']} (Score: {row['final_score']:.4f})")
        print(f"    URL: {row['url']}")
        print(f"    Summary: {row['summary'][:300]}...\n")


if __name__ == "__main__":
    main()