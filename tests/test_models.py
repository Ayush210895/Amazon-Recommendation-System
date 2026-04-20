import inspect

import pandas as pd

import amazon_recommender.models as models_module
from amazon_recommender.models import (
    ItemKNNRecommender,
    PopularityRecommender,
    SVDRecommender,
    UserKNNRecommender,
)


def _interactions() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {"user_id": 0, "item_id": 0, "rating": 5.0},
            {"user_id": 0, "item_id": 1, "rating": 4.0},
            {"user_id": 1, "item_id": 0, "rating": 5.0},
            {"user_id": 1, "item_id": 2, "rating": 2.0},
            {"user_id": 2, "item_id": 1, "rating": 4.0},
            {"user_id": 2, "item_id": 2, "rating": 1.0},
        ]
    )
    frame.attrs["n_users"] = 3
    frame.attrs["n_items"] = 3
    return frame


def test_models_predict_inside_rating_scale() -> None:
    interactions = _interactions()
    recommenders = [
        PopularityRecommender(),
        UserKNNRecommender(k=2),
        ItemKNNRecommender(k=2),
        SVDRecommender(n_factors=2),
    ]

    for recommender in recommenders:
        recommender.fit(interactions)
        prediction = recommender.predict(user_id=0, item_id=2)
        assert 1.0 <= prediction <= 5.0


def test_recommendations_do_not_include_seen_items() -> None:
    recommender = PopularityRecommender().fit(_interactions())

    recommendations = recommender.recommend(user_id=0, n=3)

    assert {rec.item_id for rec in recommendations}.isdisjoint({0, 1})


def test_core_models_do_not_import_recommender_libraries() -> None:
    source = inspect.getsource(models_module)

    forbidden_imports = ["sklearn", "surprise", "pyspark"]
    assert all(import_name not in source for import_name in forbidden_imports)
