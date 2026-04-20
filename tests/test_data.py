import pandas as pd

from amazon_recommender.data import deduplicate_reviews, encode_interactions


def test_deduplicate_reviews_keeps_latest_rating() -> None:
    reviews = pd.DataFrame(
        [
            {"reviewerID": "u1", "asin": "p1", "overall": 2.0, "unixReviewTime": 1},
            {"reviewerID": "u1", "asin": "p1", "overall": 5.0, "unixReviewTime": 2},
            {"reviewerID": "u2", "asin": "p1", "overall": 4.0, "unixReviewTime": 1},
        ]
    )

    result = deduplicate_reviews(reviews)

    assert len(result) == 2
    assert result.loc[result["reviewerID"] == "u1", "overall"].item() == 5.0


def test_encode_interactions_adds_stable_integer_ids() -> None:
    reviews = pd.DataFrame(
        [
            {"reviewerID": "u2", "asin": "p2", "overall": 4.0, "unixReviewTime": 1},
            {"reviewerID": "u1", "asin": "p1", "overall": 5.0, "unixReviewTime": 1},
        ]
    )

    encoded = encode_interactions(reviews)

    assert encoded.user_encoder == {"u1": 0, "u2": 1}
    assert encoded.item_encoder == {"p1": 0, "p2": 1}
    assert encoded.interactions.attrs["n_users"] == 2
    assert encoded.interactions.attrs["n_items"] == 2
