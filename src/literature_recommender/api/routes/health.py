from fastapi import APIRouter, Depends, status, HTTPException
from literature_recommender.api.dependencies import get_engine
from literature_recommender.recommender import LiteratureRecommender


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)

@router.get("/info",
            status_code=status.HTTP_200_OK,
            description="Get the health status of the API service.")
def health_check():
    """
    Endpoint to check the health status of the API service.
    """
    return {
        "status": "healthy",
        "service": "literature-recommender-api"
    }

@router.get("/benchmark",
            status_code=status.HTTP_200_OK,
            description="Run a benchmark test on the recommender engine.")
def run_benchmark(engine: LiteratureRecommender = Depends(get_engine)):
    """
    Endpoint to run a benchmark test on the recommender engine.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Benchmarking is not yet implemented.",
    )
    
