"""Helpers for turning model output into readable product recommendations."""

from __future__ import annotations

import pandas as pd

from amazon_recommender.models import Recommendation


def recommendations_to_frame(
    recommendations: list[Recommendation],
    item_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Attach ASIN/title metadata to recommendation IDs."""

    recs = pd.DataFrame(
        [{"item_id": rec.item_id, "score": rec.score} for rec in recommendations]
    )
    if recs.empty:
        return pd.DataFrame(columns=["item_id", "score", "asin", "title"])

    columns = [
        column
        for column in ["item_id", "asin", "title", "brand", "category"]
        if column in item_lookup.columns
    ]
    return recs.merge(item_lookup[columns], on="item_id", how="left")
