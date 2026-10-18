#!/usr/bin/env python
"""Train and evaluate custom Amazon recommendation models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amazon_recommender.data import (
    encode_interactions,
    load_metadata,
    load_reviews,
    temporal_train_test_split_interactions,
    train_test_split_interactions,
)
from amazon_recommender.metrics import format_metrics, regression_metrics
from amazon_recommender.models import (
    ItemKNNRecommender,
    PopularityRecommender,
    SVDRecommender,
    UserKNNRecommender,
)
from amazon_recommender.reporting import recommendations_to_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews-path", type=Path, default=Path("data/raw/All_Beauty_5.json.gz"))
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/raw/meta_All_Beauty.json.gz"),
    )
    parser.add_argument("--limit-reviews", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        choices=("temporal", "random"),
        default="temporal",
        help="Use a future-like temporal holdout by default.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/evaluation.json"),
    )
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    if not args.reviews_path.exists():
        raise FileNotFoundError(
            f"Missing {args.reviews_path}. Run: python scripts/download_data.py"
        )

    reviews = load_reviews(args.reviews_path, limit=args.limit_reviews)
    metadata = None
    if args.metadata_path.exists():
        metadata = load_metadata(args.metadata_path, asin_filter=set(reviews["asin"]))

    encoded = encode_interactions(reviews, metadata=metadata)
    if args.split_strategy == "temporal":
        train, test = temporal_train_test_split_interactions(
            encoded.interactions, test_size=args.test_size
        )
    else:
        train, test = train_test_split_interactions(
            encoded.interactions,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    models = {
        "popularity": PopularityRecommender(min_rating_count=1, prior_weight=5.0),
        "user_knn": UserKNNRecommender(k=20),
        "item_knn": ItemKNNRecommender(k=20),
        "svd": SVDRecommender(n_factors=20),
    }

    print(
        "Dataset:",
        f"{len(encoded.interactions):,} ratings,",
        f"{len(encoded.user_encoder):,} users,",
        f"{len(encoded.item_encoder):,} products",
    )
    print(f"Train/test split: {len(train):,}/{len(test):,}")
    print()

    model_metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(train)
        metrics = regression_metrics(model, test)
        model_metrics[name] = metrics
        print(f"{name}: {format_metrics(metrics)}")

    evaluation = {
        "split_strategy": args.split_strategy,
        "test_size": args.test_size,
        "ratings": len(encoded.interactions),
        "users": len(encoded.user_encoder),
        "products": len(encoded.item_encoder),
        "train_rows": len(train),
        "test_rows": len(test),
        "models": model_metrics,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Wrote evaluation evidence: {args.metrics_output}")

    first_user_id = int(encoded.interactions["user_id"].iloc[0])
    print(f"\nTop {args.top_n} SVD recommendations for user_id={first_user_id}:")
    recommendation_frame = recommendations_to_frame(
        models["svd"].recommend(first_user_id, n=args.top_n),
        encoded.item_lookup,
    )
    display_columns = [
        column
        for column in ["asin", "title", "brand", "score"]
        if column in recommendation_frame.columns
    ]
    print(recommendation_frame[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
