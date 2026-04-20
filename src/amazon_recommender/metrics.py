"""Evaluation metrics for recommendation experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(model: object, test: pd.DataFrame) -> dict[str, float]:
    """Evaluate rating predictions on held-out user-item interactions."""

    actual: list[float] = []
    predicted: list[float] = []

    for row in test.itertuples(index=False):
        try:
            prediction = model.predict(int(row.user_id), int(row.item_id))
        except IndexError:
            continue
        actual.append(float(row.rating))
        predicted.append(float(prediction))

    if not actual:
        return {"mae": float("nan"), "rmse": float("nan"), "coverage": 0.0}

    y_true = np.asarray(actual)
    y_pred = np.asarray(predicted)
    errors = y_true - y_pred
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "coverage": float(len(actual) / len(test)),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
