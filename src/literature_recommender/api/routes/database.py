from typing import Annotated

from fastapi import APIRouter, Depends, status

from literature_recommender.api.dependencies import get_engine
from literature_recommender.api.schemas import DatabaseStatusResponse, ReindexResponse
from literature_recommender.recommender import LiteratureRecommender

router = APIRouter(
    prefix="/database",
    tags=["Database"],
)

@router.get(
    "/status", 
    response_model=DatabaseStatusResponse,
    status_code=status.HTTP_200_OK,
    description="Get the current status of the database, including the number of indexed articles and pending articles.",
)
def get_database_status(engine: Annotated[LiteratureRecommender, Depends(get_engine)]):
    """
    Endpoint to retrieve the current status of the database, including the number of indexed articles and pending articles.
    """
    return DatabaseStatusResponse(
        indexed_articles=len(engine.df),
        pending_articles=engine.get_pending_count(),
    )


@router.post(
    "/reindex",
    response_model=ReindexResponse,
    status_code=status.HTTP_200_OK,
    description="Trigger a re-indexing of the database.",
)
def trigger_reindex(engine: Annotated[LiteratureRecommender, Depends(get_engine)]):
    """
    Endpoint to trigger a re-indexing of the database.
    """
    added = engine.reindex_database()
    return ReindexResponse(
        articles_added=added,
        total_indexed=len(engine.df),
        message=f"Successfully re-indexed database. Added {added} new articles.",
    )