from .database import router as database_router
from .health import router as health_router
from .recommender import router as recommender_router

__all__ = ["database_router", "recommender_router", "health_router"]