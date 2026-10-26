from pydantic import BaseModel, Field, HttpUrl

class RecommendationRequest(BaseModel):
    """
    Schema for the request for article recommendations.
    """
    urls: list[HttpUrl] = Field(...,
                                min_length=1, 
                                description="List of Wikipedia URLs used to generate recommendations.",
                                example=["https://en.wikipedia.org/wiki/Franz_Kafka"]
    )
    top_k: int = Field(default=5, 
                       ge=1, 
                       le=10, 
                       description="Number of top recommendations to return."
    )
    method: str = Field(default="hybrid",
                        pattern="^(hybrid|dense|tfidf)$",
                        description="Final scoring method."
    )
    alpha: float = Field(default=0.7,
                        ge=0.0,
                        le=1.0,
                        description="Weight for the hybrid method."
    )


class ArticleRecommendation(BaseModel):
    """
    Schema for a single article recommendation.
    """
    title: str = Field(..., description="Title of the recommended article.")
    url: HttpUrl = Field(..., description="URL of the recommended article.")
    final_score: float = Field(..., description="Relevance score of the recommended article.")
    dense_score: float = Field(..., description="Relevance score of the recommended article.")
    tfidf_score: float = Field(..., description="Relevance score of the recommended article.")
    summary: str = Field(..., description="Summary of the recommended article.")


class RecommendationResponse(BaseModel):
    """
    Schema for the response with article recommendations.
    """
    query_urls: list[HttpUrl] = Field(..., description="List of Wikipedia URLs used to generate recommendations.")
    method: str = Field(..., description="Final scoring method used for recommendations.")
    count: int = Field(..., description="Number of recommendations returned.")
    recommendations: list[ArticleRecommendation] = Field(..., description="List of recommended articles.")


class DatabaseStatusResponse(BaseModel):
    """
    Schema for the response indicating count of indexed and pending articles in the database.
    """
    indexed_articles: int = Field(..., description="Number of articles indexed in the database.")
    pending_articles: int = Field(..., description="Number of articles pending to be indexed in the database.")


class ReindexResponse(BaseModel):
    """
    Schema for the response indicating the status of the reindexing process.
    """
    articles_added: int = Field(..., description="Number of articles added to the database during reindexing.")
    total_indexed: int = Field(..., description="Total number of articles indexed in the database after reindexing.")
    message: str = Field(..., description="Message indicating the status of the reindexing process.")