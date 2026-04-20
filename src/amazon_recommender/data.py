"""Data loading and preparation helpers for Amazon review recommendations."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REVIEW_COLUMNS = [
    "reviewerID",
    "asin",
    "overall",
    "unixReviewTime",
    "reviewTime",
    "summary",
    "reviewText",
    "verified",
]

METADATA_COLUMNS = [
    "asin",
    "title",
    "brand",
    "price",
    "category",
    "description",
]


@dataclass(frozen=True)
class EncodedInteractions:
    """Interaction table plus the ID maps needed to decode recommendations."""

    interactions: pd.DataFrame
    user_encoder: dict[str, int]
    item_encoder: dict[str, int]
    item_lookup: pd.DataFrame

    @property
    def user_decoder(self) -> dict[int, str]:
        return {value: key for key, value in self.user_encoder.items()}

    @property
    def item_decoder(self) -> dict[int, str]:
        return {value: key for key, value in self.item_encoder.items()}


def iter_json_gz(path: str | Path, limit: int | None = None) -> Iterable[dict]:
    """Yield records from a gzip-compressed JSON-lines file."""

    with gzip.open(path, "rt", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield json.loads(line)


def load_reviews(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Load Amazon reviews and keep the fields used by the recommenders."""

    reviews = pd.DataFrame(iter_json_gz(path, limit=limit))
    if reviews.empty:
        return pd.DataFrame(columns=REVIEW_COLUMNS)

    for column in REVIEW_COLUMNS:
        if column not in reviews:
            reviews[column] = pd.NA

    reviews = reviews[REVIEW_COLUMNS].copy()
    reviews["overall"] = pd.to_numeric(reviews["overall"], errors="coerce")
    reviews["unixReviewTime"] = pd.to_numeric(
        reviews["unixReviewTime"], errors="coerce"
    ).fillna(0)
    reviews = reviews.dropna(subset=["reviewerID", "asin", "overall"])
    return reviews


def load_metadata(
    path: str | Path,
    asin_filter: set[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load product metadata and optionally keep only products in the reviews."""

    rows: list[dict] = []
    for record in iter_json_gz(path, limit=limit):
        asin = record.get("asin")
        if asin_filter is not None and asin not in asin_filter:
            continue
        rows.append(
            {
                "asin": asin,
                "title": record.get("title") or asin,
                "brand": record.get("brand"),
                "price": record.get("price"),
                "category": _first_category(record.get("category")),
                "description": _first_text(record.get("description")),
            }
        )

    metadata = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    return metadata.dropna(subset=["asin"]).drop_duplicates("asin", keep="last")


def deduplicate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest rating for duplicate user/product pairs."""

    ordered = reviews.sort_values(
        ["reviewerID", "asin", "unixReviewTime"],
        ascending=[True, True, True],
    )
    return ordered.drop_duplicates(["reviewerID", "asin"], keep="last").reset_index(
        drop=True
    )


def encode_interactions(
    reviews: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> EncodedInteractions:
    """Convert string reviewer/product IDs into compact integer IDs."""

    clean_reviews = deduplicate_reviews(reviews)
    reviewer_ids = sorted(clean_reviews["reviewerID"].astype(str).unique())
    item_ids = sorted(clean_reviews["asin"].astype(str).unique())

    user_encoder = {reviewer_id: index for index, reviewer_id in enumerate(reviewer_ids)}
    item_encoder = {asin: index for index, asin in enumerate(item_ids)}

    interactions = clean_reviews.copy()
    interactions["user_id"] = interactions["reviewerID"].astype(str).map(user_encoder)
    interactions["item_id"] = interactions["asin"].astype(str).map(item_encoder)
    interactions["rating"] = interactions["overall"].astype(float)
    interactions = interactions[
        ["user_id", "item_id", "rating", "reviewerID", "asin", "unixReviewTime"]
    ].reset_index(drop=True)
    interactions.attrs["n_users"] = len(user_encoder)
    interactions.attrs["n_items"] = len(item_encoder)

    item_lookup = pd.DataFrame({"asin": item_ids})
    item_lookup["item_id"] = item_lookup["asin"].map(item_encoder)
    if metadata is not None and not metadata.empty:
        item_lookup = item_lookup.merge(metadata, on="asin", how="left")

    if "title" not in item_lookup:
        item_lookup["title"] = item_lookup["asin"]
    item_lookup["title"] = item_lookup["title"].fillna(item_lookup["asin"])
    return EncodedInteractions(
        interactions=interactions,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        item_lookup=item_lookup,
    )


def train_test_split_interactions(
    interactions: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible random interaction split."""

    test = interactions.sample(frac=test_size, random_state=random_state)
    train = interactions.drop(test.index)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.attrs.update(interactions.attrs)
    test.attrs.update(interactions.attrs)
    return train, test


def _first_category(value: object) -> str | None:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, list):
            return " > ".join(str(part) for part in first)
        return str(first)
    return None


def _first_text(value: object) -> str | None:
    if isinstance(value, list):
        return " ".join(str(part) for part in value if part)
    if value is None:
        return None
    return str(value)
