import logging

logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(
    logging.ERROR
)

import streamlit as st
import pandas as pd
from literature_recommender import LiteratureRecommender, RecommendationVisualizer


st.set_page_config(
    page_title="Literature Recommender",
    layout="wide",
)

@st.cache_resource
def get_engine():
    return LiteratureRecommender(load_precomputed=True)


engine = get_engine()
visualizer = RecommendationVisualizer(engine)


st.sidebar.title("Engine Controls")

method = st.sidebar.selectbox(
    "Retrieval Method",
    options=["hybrid", "dense", "tfidf"],
    index=0,
    help="Hybrid combines semantic embeddings with exact keyword matches.",
)

top_k = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

alpha = 0.70
if method == "hybrid":
    alpha = st.sidebar.slider(
        "Semantic Weight (Alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        help="Higher alpha emphasizes latent literary themes; lower emphasizes exact keyword overlap.",
    )

st.sidebar.divider()
st.sidebar.subheader("Database Status")
indexed_count = len(engine.df)
pending_count = engine.get_pending_count()

st.sidebar.metric("Indexed Articles", indexed_count)
st.sidebar.metric("Pending Unseen Articles", pending_count)

if pending_count > 0:
    st.sidebar.info(f"{pending_count} article(s) ingested from recent queries.")
    if st.sidebar.button("Sync & Re-index Database", type="primary"):
        with st.spinner("Updating database and representations..."):
            added = engine.reindex_database()
            st.cache_resource.clear()
            st.sidebar.success(f"Added {added} new articles to the index!")
            st.rerun()


st.title("Classic Literature Recommender")
st.caption(
    "Content-based retrieval system using dual sparse TF-IDF and dense Sentence-Transformer representations."
)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Sample: Franz Kafka"):
        st.session_state["query_input"] = "https://en.wikipedia.org/wiki/Franz_Kafka"
with col2:
    if st.button("Sample: Sartre & Camus (Existentialism)"):
        st.session_state["query_input"] = (
            "https://en.wikipedia.org/wiki/Jean-Paul_Sartre\nhttps://en.wikipedia.org/wiki/Albert_Camus"
        )
with col3:
    if st.button("Sample: Italo Calvino (Unseen Book Test)"):
        st.session_state["query_input"] = (
            "https://en.wikipedia.org/wiki/Italo_Calvino\nhttps://en.wikipedia.org/wiki/Invisible_Cities"
        )

default_text = st.session_state.get(
    "query_input", "https://en.wikipedia.org/wiki/Franz_Kafka"
)
raw_input = st.text_area(
    "Enter Wikipedia Article URLs (one per line):",
    value=default_text,
    height=110,
)

urls = [line.strip() for line in raw_input.strip().split("\n") if line.strip()]

if st.button("Recommend Articles", type="primary"):
    if not urls:
        st.warning("Please provide at least one valid Wikipedia URL.")
    else:
        with st.spinner("Scraping unseen links and computing user profile..."):
            try:
                prev_pending = engine.get_pending_count()

                recs_df = engine.recommend_articles(
                    visited_links=urls,
                    top_k=top_k,
                    method=method,
                    alpha=alpha,
                )
                st.session_state["last_recs"] = recs_df
                st.session_state["last_urls"] = urls

                new_pending = engine.get_pending_count()
                if new_pending > prev_pending:
                    st.toast(
                        f"Ingested {new_pending - prev_pending} unseen article(s) into pending queue!"
                    )
                    st.rerun()
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

if "last_recs" in st.session_state:
    recs = st.session_state["last_recs"]
    query_urls = st.session_state["last_urls"]

    st.subheader(f"Top {len(recs)} Recommendations")

    for idx, row in recs.reset_index(drop=True).iterrows():
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
        st.markdown(
            "#### Word-Level Attribution ($TFIDF_q \\times TFIDF_r$)"
        )
        rec_choice = st.selectbox(
            "Select recommended book to inspect:",
            options=recs["url"].tolist(),
            format_func=lambda u: recs.loc[recs["url"] == u, "title"].values[0],
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
            fig_decomp = visualizer.plot_score_decomposition(
                url=query_urls[0],
                top_k=len(recs),
                method=method,
                show=False,
            )
            st.pyplot(fig_decomp)

    with tab3:
        st.markdown("#### Profile Component Influence")
        if len(query_urls) > 1:
            target_choice = st.selectbox(
                "Select target to attribute:",
                options=recs["url"].tolist(),
                format_func=lambda u: recs.loc[recs["url"] == u, "title"].values[0],
                key="attr_target",
            )
            fig_attr = visualizer.plot_user_profile_attribution(
                visited_urls=query_urls,
                recommended_target_url=target_choice,
                show=False,
            )
            st.pyplot(fig_attr)
        else:
            st.info("Input at least two URLs to evaluate multi-article influence.")