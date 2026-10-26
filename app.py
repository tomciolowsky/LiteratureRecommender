import logging
import pandas as pd
import streamlit as st
import os
import httpx
from literature_recommender import LiteratureRecommender, RecommendationVisualizer
from fastapi import status

logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(
    logging.ERROR
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

@st.cache_resource
def get_visualizer():
    """
    Get and cache the LiteratureRecommender engine and RecommendationVisualizer.
    """
    engine = LiteratureRecommender()
    return engine, RecommendationVisualizer(engine)


class RecommenderAPIClient:
    """
    Client for interacting with the Literature Recommender API.
    """
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> bool:
        """
        Check the health of the API.
        """
        try:
            response = httpx.get(f"{self.base_url}/health/info", timeout=3.0)
            return response.status_code == status.HTTP_200_OK
        except httpx.RequestError as e:
            logging.error(f"API health check failed: {e}")
            return False

    def get_database_status(self) -> dict:
        """
        Get the status of the database from the API.
        """
        try:
            response = httpx.get(f"{self.base_url}/database/status", timeout=3.0)
            if response.status_code == status.HTTP_200_OK:
                return response.json()
        
        except httpx.RequestError as e:
            logging.error(f"Failed to get database status: {e}")
            return {"indexed_count": 0, "pending_count": 0}

    def trigger_reindex(self) -> dict:
        """
        Trigger a reindexing of the database via the API.
        """
        response = httpx.post(f"{self.base_url}/database/reindex", timeout=30.0)
        response.raise_for_status()
        return response.json()
        

    def get_recommendations(self, urls: list[str], top_k: int, method: str, alpha: float) -> list[dict]:
        """
        Get recommendations from the API.
        """
        payload = {
            "urls": urls,
            "top_k": top_k,
            "method": method,
            "alpha": alpha,
        }
        response = httpx.post(f"{self.base_url}/recommend/", json=payload, timeout=30.0)
        if response.status_code != status.HTTP_200_OK:
            error_detail = response.json().get("detail", response.text)
            raise RuntimeError(f"API Error ({response.status_code}): {error_detail}")
        return response.json().get("recommendations", [])


api_client = RecommenderAPIClient(API_BASE_URL)

def init_session_state():
    """
    Ensures required session keys exist upon initial load.
    """
    if "recs_df" not in st.session_state:
        st.session_state["recs_df"] = None
    if "query_urls" not in st.session_state:
        st.session_state["query_urls"] = []
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = ("https://en.wikipedia.org/wiki/Franz_Kafka")


def render_sidebar(api_online: bool):
    """
    Renders retrieval hyperparameters and database management controls.
    """
    st.sidebar.title("Engine Controls")

    if api_online:
        st.sidebar.success("API Connected (`:8000`)")
    else:
        st.sidebar.error("API Offline — start FastAPI server")

    
    method = st.sidebar.selectbox(
        "Retrieval Method",
        options=["hybrid", "dense", "tfidf"],
        index=0,
        help="Hybrid combines semantic embeddings with exact keyword matches.",
    )

    top_k = st.sidebar.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5,
    )

    alpha = 0.70
    if method == "hybrid":
        alpha = st.sidebar.slider(
            "Semantic Weight (Alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            help="Higher alpha emphasizes latent literary themes; lower emphasizes exact keyword overlap."
        )

    st.sidebar.divider()
    st.sidebar.subheader("Database Status")

    db_status = api_client.get_database_status()
    st.sidebar.metric("Indexed Articles", db_status["indexed_articles"])
    st.sidebar.metric("Pending Ingestion", db_status["pending_articles"])

    if db_status["pending_articles"] > 0:
        st.sidebar.info(f"{db_status['pending_articles']} article(s) ingested from recent queries.")
        if st.sidebar.button("Sync & Re-index Database", type="primary"):
            with st.spinner("Updating database and representations..."):
                try:
                    result = api_client.trigger_reindex()
                    st.sidebar.success(result.get("message", "Database updated!"))
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Re-indexing failed: {e}")

    return method, top_k, alpha


def render_sample_buttons():
    """
    Renders the main title and preset sample buttons.
    """
    st.title("Classic Literature Recommender")
    st.caption(
        "Content-based retrieval system using dual sparse TF-IDF and dense Sentence-Transformer representations.\nPowered by **FastAPI** backend and **Streamlit** frontend."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Sample: Franz Kafka"):
            st.session_state["query_input"] = (
                "https://en.wikipedia.org/wiki/Franz_Kafka"
            )
            st.rerun()
    with col2:
        if st.button("Sample: Sartre & Camus (Existentialism)"):
            st.session_state["query_input"] = (
                "https://en.wikipedia.org/wiki/Jean-Paul_Sartre\nhttps://en.wikipedia.org/wiki/Albert_Camus"
            )
            st.rerun()
    with col3:
        if st.button("Sample: Italo Calvino (Unseen Test)"):
            st.session_state["query_input"] = (
                "https://en.wikipedia.org/wiki/Italo_Calvino"
            )
            st.rerun()


def render_query_input() -> list[str]:
    """
    Renders the query URL input area and returns cleaned URLs.
    """
    raw_input = st.text_area(
        "Enter Wikipedia Article URLs (one per line):",
        value=st.session_state["query_input"],
        height=110,
    )
    return [line.strip() for line in raw_input.strip().split("\n") if line.strip()]


def render_results(recs_df: pd.DataFrame):
    """
    Displays the retrieved recommendations as formatted cards.
    """
    st.subheader(f"Top {len(recs_df)} Recommendations")

    for idx, row in recs_df.reset_index(drop=True).iterrows():
        with st.container(border=True):
            col_info, col_scores = st.columns([3, 1])
            with col_info:
                st.markdown(f"### {idx + 1}. [{row['title']}]({row['url']})")
                st.write(row["summary"])
            with col_scores:
                st.metric("Final Score", f"{row['final_score']:.4f}")
                st.caption(
                    f"**Dense Semantic:** {row['dense_score']:.4f}\n\n**TF-IDF Lexical:** {row['tfidf_score']:.4f}"
                )


def render_visualizations(
    query_urls: list[str],
    recs_df: pd.DataFrame,
    method: str,
    visualizer: RecommendationVisualizer,
):
    """
    Renders diagnostic attribution plots in dedicated tabs.
    """
    st.divider()
    st.subheader("Visual Explanations & Diagnostics")

    tab1, tab2, tab3 = st.tabs(
        [
            "Prediction Breakdown",
            "Score Decomposition",
            "Multi-Article Attribution",
        ]
    )

    with tab1:
        st.markdown("#### Word-Level Attribution ($TFIDF_q \\times TFIDF_r$)")
        rec_choice = st.selectbox(
            "Select recommended book to inspect:",
            options=recs_df["url"].tolist(),
            format_func=lambda u: recs_df.loc[recs_df["url"] == u, "title"].values[0],
        )
        if query_urls:
            fig_breakdown = visualizer.plot_prediction_breakdown(
                query_url=query_urls[0],
                recommendation_url=rec_choice,
                show=False,
            )
            st.pyplot(fig_breakdown)

    with tab2:
        st.markdown("#### Dense vs. TF-IDF vs. Final Scores")
        if query_urls:
            fig_decomposition = visualizer.plot_score_decomposition(
                url=query_urls[0],
                top_k=len(recs_df),
                method=method,
                show=False,
            )
            st.pyplot(fig_decomposition)

    with tab3:
        st.markdown("#### Profile Component Influence")
        if len(query_urls) > 1:
            target_choice = st.selectbox(
                "Select target to attribute:",
                options=recs_df["url"].tolist(),
                format_func=lambda u: recs_df.loc[recs_df["url"] == u, "title"].values[0],
                key="tab3_target",
            )
            fig_attribution = visualizer.plot_user_profile_attribution(
                visited_urls=query_urls,
                recommended_target_url=target_choice,
                show=False,
            )
            st.pyplot(fig_attribution)
        else:
            st.info("Input at least two URLs to evaluate multi-article influence.")


def main():
    """
    Main function to run the Streamlit app, handling user input, API calls, and rendering results.
    """
    st.set_page_config(
        page_title="Literature Recommender",
        layout="wide"
    )

    init_session_state()
    api_online = api_client.check_health()
    _, visualizer = get_visualizer()

    method, top_k, alpha = render_sidebar(api_online)
    render_sample_buttons()
    urls = render_query_input()

    if st.button("Recommend Articles", type="primary", disabled=(not api_online)):
        if not urls:
            st.warning("Please provide at least one valid Wikipedia URL.")
            return

        with st.spinner("Calling FastAPI service to compute profile & retrieve matches..."):
            try:
                raw_recs = api_client.get_recommendations(
                    urls=urls,
                    top_k=top_k,
                    method=method,
                    alpha=alpha,
                )
                st.session_state["recs_df"] = pd.DataFrame(raw_recs)
                st.session_state["query_urls"] = urls
                st.rerun()
            except Exception as e:
                st.error(f"Recommendation request failed: {e}")

    if (st.session_state["recs_df"] is not None) and (not st.session_state["recs_df"].empty):
        render_results(st.session_state["recs_df"])
        render_visualizations(
            query_urls=st.session_state["query_urls"],
            recs_df=st.session_state["recs_df"],
            method=method,
            visualizer=visualizer,
        )


if __name__ == "__main__":
    main()
