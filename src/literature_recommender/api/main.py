from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from literature_recommender.api.routes import (
    database_router,
    health_router,
    recommender_router,
)
from literature_recommender.recommender import LiteratureRecommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = LiteratureRecommender()
    yield
    app.state.engine = None

app = FastAPI(
    title="Literature Recommender API",
    description="An API for recommending literature articles based on Wikipedia URLs.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(database_router)
app.include_router(health_router)
app.include_router(recommender_router)