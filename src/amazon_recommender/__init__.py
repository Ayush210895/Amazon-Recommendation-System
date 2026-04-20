"""Amazon product recommendation package."""

from amazon_recommender.models import (
    ItemKNNRecommender,
    PopularityRecommender,
    SVDRecommender,
    UserKNNRecommender,
)

__all__ = [
    "ItemKNNRecommender",
    "PopularityRecommender",
    "SVDRecommender",
    "UserKNNRecommender",
]
