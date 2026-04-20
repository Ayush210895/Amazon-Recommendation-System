"""Custom recommendation models built with NumPy.

The original notebook explored ALS, KNN, and SVD ideas with external modeling
packages. This module keeps the educational model logic in the repository by
implementing the core scoring steps directly with NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ArrayLike = np.ndarray


@dataclass(frozen=True)
class Recommendation:
    item_id: int
    score: float


class PopularityRecommender:
    """Recommend globally strong products using a Bayesian average rating."""

    def __init__(self, min_rating_count: int = 1, prior_weight: float = 5.0) -> None:
        self.min_rating_count = min_rating_count
        self.prior_weight = prior_weight
        self.global_mean_: float = 0.0
        self.item_scores_: dict[int, float] = {}
        self.seen_items_: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        ratings = interactions["rating"].astype(float)
        self.global_mean_ = float(ratings.mean())
        grouped = interactions.groupby("item_id")["rating"].agg(["mean", "count"])
        grouped = grouped[grouped["count"] >= self.min_rating_count]
        score = (
            grouped["count"] * grouped["mean"] + self.prior_weight * self.global_mean_
        ) / (grouped["count"] + self.prior_weight)
        self.item_scores_ = score.to_dict()
        self.seen_items_ = _seen_items(interactions)
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        return float(self.item_scores_.get(item_id, self.global_mean_))

    def recommend(self, user_id: int, n: int = 10) -> list[Recommendation]:
        seen = self.seen_items_.get(user_id, set())
        ranked = sorted(
            (
                Recommendation(int(item_id), float(score))
                for item_id, score in self.item_scores_.items()
                if item_id not in seen
            ),
            key=lambda rec: rec.score,
            reverse=True,
        )
        return ranked[:n]


class UserKNNRecommender:
    """User-based collaborative filtering with mean-centered cosine similarity."""

    def __init__(self, k: int = 20) -> None:
        self.k = k
        self.global_mean_: float = 0.0
        self.user_means_: ArrayLike | None = None
        self.rating_matrix_: ArrayLike | None = None
        self.similarity_: ArrayLike | None = None
        self.seen_items_: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame) -> "UserKNNRecommender":
        matrix = _build_rating_matrix(interactions)
        self.rating_matrix_ = matrix
        self.global_mean_ = _safe_mean(matrix[matrix > 0])
        self.user_means_ = _row_means(matrix, self.global_mean_)
        centered = _mean_center_rows(matrix, self.user_means_)
        self.similarity_ = _cosine_similarity(centered)
        self.seen_items_ = _seen_items(interactions)
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        self._check_fit()
        matrix = self.rating_matrix_
        means = self.user_means_
        similarity = self.similarity_
        assert matrix is not None and means is not None and similarity is not None

        item_ratings = matrix[:, item_id]
        rated_mask = item_ratings > 0
        rated_mask[user_id] = False
        neighbor_ids = np.where(rated_mask)[0]
        if len(neighbor_ids) == 0:
            return float(means[user_id])

        weights = similarity[user_id, neighbor_ids]
        top_positions = _top_abs_indexes(weights, self.k)
        neighbor_ids = neighbor_ids[top_positions]
        weights = weights[top_positions]
        denominator = np.sum(np.abs(weights))
        if denominator == 0:
            return float(means[user_id])

        deviations = matrix[neighbor_ids, item_id] - means[neighbor_ids]
        prediction = means[user_id] + np.dot(weights, deviations) / denominator
        return _clip_rating(prediction)

    def recommend(self, user_id: int, n: int = 10) -> list[Recommendation]:
        self._check_fit()
        matrix = self.rating_matrix_
        assert matrix is not None

        seen = self.seen_items_.get(user_id, set())
        candidates = [item_id for item_id in range(matrix.shape[1]) if item_id not in seen]
        ranked = [
            Recommendation(item_id, self.predict(user_id, item_id))
            for item_id in candidates
        ]
        ranked.sort(key=lambda rec: rec.score, reverse=True)
        return ranked[:n]

    def _check_fit(self) -> None:
        if self.rating_matrix_ is None or self.similarity_ is None:
            raise RuntimeError("Call fit before predict or recommend.")


class ItemKNNRecommender:
    """Item-based collaborative filtering with adjusted cosine similarity."""

    def __init__(self, k: int = 20) -> None:
        self.k = k
        self.global_mean_: float = 0.0
        self.item_means_: ArrayLike | None = None
        self.rating_matrix_: ArrayLike | None = None
        self.similarity_: ArrayLike | None = None
        self.seen_items_: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame) -> "ItemKNNRecommender":
        matrix = _build_rating_matrix(interactions)
        self.rating_matrix_ = matrix
        self.global_mean_ = _safe_mean(matrix[matrix > 0])
        self.item_means_ = _column_means(matrix, self.global_mean_)
        centered = matrix.copy()
        observed = centered > 0
        centered[observed] = centered[observed] - np.take(
            self.item_means_, np.where(observed)[1]
        )
        self.similarity_ = _cosine_similarity(centered.T)
        self.seen_items_ = _seen_items(interactions)
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        self._check_fit()
        matrix = self.rating_matrix_
        means = self.item_means_
        similarity = self.similarity_
        assert matrix is not None and means is not None and similarity is not None

        user_ratings = matrix[user_id, :]
        rated_items = np.where(user_ratings > 0)[0]
        if len(rated_items) == 0:
            return float(means[item_id])

        weights = similarity[item_id, rated_items]
        top_positions = _top_abs_indexes(weights, self.k)
        rated_items = rated_items[top_positions]
        weights = weights[top_positions]
        denominator = np.sum(np.abs(weights))
        if denominator == 0:
            return float(means[item_id])

        deviations = user_ratings[rated_items] - means[rated_items]
        prediction = means[item_id] + np.dot(weights, deviations) / denominator
        return _clip_rating(prediction)

    def recommend(self, user_id: int, n: int = 10) -> list[Recommendation]:
        self._check_fit()
        matrix = self.rating_matrix_
        assert matrix is not None

        seen = self.seen_items_.get(user_id, set())
        candidates = [item_id for item_id in range(matrix.shape[1]) if item_id not in seen]
        ranked = [
            Recommendation(item_id, self.predict(user_id, item_id))
            for item_id in candidates
        ]
        ranked.sort(key=lambda rec: rec.score, reverse=True)
        return ranked[:n]

    def _check_fit(self) -> None:
        if self.rating_matrix_ is None or self.similarity_ is None:
            raise RuntimeError("Call fit before predict or recommend.")


class SVDRecommender:
    """Low-rank matrix factorization using NumPy's singular value decomposition."""

    def __init__(self, n_factors: int = 20) -> None:
        self.n_factors = n_factors
        self.global_mean_: float = 0.0
        self.reconstructed_: ArrayLike | None = None
        self.seen_items_: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame) -> "SVDRecommender":
        matrix = _build_rating_matrix(interactions)
        self.global_mean_ = _safe_mean(matrix[matrix > 0])
        user_means = _row_means(matrix, self.global_mean_)
        filled = matrix.copy()
        missing = filled == 0
        filled[missing] = np.take(user_means, np.where(missing)[0])

        centered = filled - user_means[:, None]
        u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        rank = min(self.n_factors, len(singular_values))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            low_rank = (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]
        low_rank = np.nan_to_num(low_rank, nan=0.0, posinf=5.0, neginf=-5.0)
        self.reconstructed_ = low_rank + user_means[:, None]
        self.seen_items_ = _seen_items(interactions)
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        if self.reconstructed_ is None:
            raise RuntimeError("Call fit before predict or recommend.")
        return _clip_rating(self.reconstructed_[user_id, item_id])

    def recommend(self, user_id: int, n: int = 10) -> list[Recommendation]:
        if self.reconstructed_ is None:
            raise RuntimeError("Call fit before predict or recommend.")
        seen = self.seen_items_.get(user_id, set())
        ranked = [
            Recommendation(item_id, _clip_rating(score))
            for item_id, score in enumerate(self.reconstructed_[user_id])
            if item_id not in seen
        ]
        ranked.sort(key=lambda rec: rec.score, reverse=True)
        return ranked[:n]


def _build_rating_matrix(interactions: pd.DataFrame) -> ArrayLike:
    n_users = int(interactions.attrs.get("n_users", interactions["user_id"].max() + 1))
    n_items = int(interactions.attrs.get("n_items", interactions["item_id"].max() + 1))
    matrix = np.zeros((n_users, n_items), dtype=float)
    matrix[
        interactions["user_id"].to_numpy(dtype=int),
        interactions["item_id"].to_numpy(dtype=int),
    ] = interactions["rating"].to_numpy(dtype=float)
    return matrix


def _seen_items(interactions: pd.DataFrame) -> dict[int, set[int]]:
    grouped = interactions.groupby("user_id")["item_id"].apply(lambda values: set(values))
    return {int(user_id): {int(item_id) for item_id in item_ids} for user_id, item_ids in grouped.items()}


def _safe_mean(values: ArrayLike) -> float:
    return float(values.mean()) if values.size else 0.0


def _row_means(matrix: ArrayLike, fallback: float) -> ArrayLike:
    counts = (matrix > 0).sum(axis=1)
    sums = matrix.sum(axis=1)
    return np.divide(sums, counts, out=np.full(matrix.shape[0], fallback), where=counts > 0)


def _column_means(matrix: ArrayLike, fallback: float) -> ArrayLike:
    counts = (matrix > 0).sum(axis=0)
    sums = matrix.sum(axis=0)
    return np.divide(sums, counts, out=np.full(matrix.shape[1], fallback), where=counts > 0)


def _mean_center_rows(matrix: ArrayLike, row_means: ArrayLike) -> ArrayLike:
    centered = matrix.copy()
    observed = centered > 0
    centered[observed] = centered[observed] - np.take(row_means, np.where(observed)[0])
    return centered


def _cosine_similarity(matrix: ArrayLike) -> ArrayLike:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator = norms @ norms.T
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        numerator = matrix @ matrix.T
    numerator = np.nan_to_num(numerator, nan=0.0, posinf=0.0, neginf=0.0)
    similarity = np.divide(
        numerator,
        denominator,
        out=np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float),
        where=denominator > 0,
    )
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _top_abs_indexes(values: ArrayLike, k: int) -> ArrayLike:
    if len(values) <= k:
        return np.arange(len(values))
    return np.argsort(np.abs(values))[-k:]


def _clip_rating(value: float) -> float:
    return float(np.clip(value, 1.0, 5.0))
