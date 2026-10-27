from fastapi import HTTPException, Request, status

from literature_recommender.recommender import LiteratureRecommender


class EngineNotFoundException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommender engine is not found.",
        )


def get_engine(request: Request) -> LiteratureRecommender:
    """
    Dependency to extract the active engine from application state.
    Raises EngineNotFoundException if the engine is not found.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise EngineNotFoundException()
    return engine