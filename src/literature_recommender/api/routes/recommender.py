from fastapi import APIRouter, Depends, HTTPException, status
from literature_recommender.api.schemas import RecommendationRequest, RecommendationResponse
from literature_recommender.api.dependencies import get_engine
from literature_recommender.recommender import LiteratureRecommender


router = APIRouter(
    prefix="/recommend",
    tags=["Recommender"],
)

@router.post("/",
             response_model=RecommendationResponse,
             status_code=status.HTTP_200_OK,
             description="Recommend literature articles based on visited Wikipedia URLs.")
def recommend_articles(
    payload: RecommendationRequest, 
    engine: LiteratureRecommender = Depends(get_engine)
):
    """
    Endpoint to recommend literature articles based on visited Wikipedia URLs.
    """
    try:
        recs_df = engine.recommend_articles(
            visited_links=[str(url) for url in payload.urls],
            top_k=payload.top_k,
            method=payload.method,
            alpha=payload.alpha
        )

        recs_dict = recs_df.to_dict(orient="records")
        return RecommendationResponse(
            query_urls=payload.urls,
            method=payload.method,
            count=len(recs_dict),
            recommendations=recs_dict,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )