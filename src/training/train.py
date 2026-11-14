import json
import logging
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
import yaml

from src.features.feature_store import RecommendationFeatureStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PopularityRecommender:
    """Baseline popularity recommender using Bayesian smoothed item ratings."""

    def __init__(self):
        self.ranked_items = []
        self.item_score_map = {}
        self.user_bias_map = {}
        self.global_mean = 3.5

    def fit(self, item_features: pd.DataFrame, user_features: pd.DataFrame, global_mean: float):
        self.global_mean = global_mean
        sorted_items = item_features.sort_values(
            by=["item_popularity_count", "item_bayesian_rating"], ascending=False
        )
        self.ranked_items = sorted_items["item_id"].astype(int).tolist()
        self.item_score_map = dict(
            zip(sorted_items["item_id"].astype(int), sorted_items["item_bayesian_rating"].astype(float))
        )
        self.user_bias_map = dict(
            zip(
                user_features["user_id"].astype(int),
                (user_features["user_avg_rating"] - global_mean).astype(float),
            )
        )
        return self

    def predict_rating(self, user_id: int, item_id: int) -> float:
        base = self.item_score_map.get(int(item_id), self.global_mean)
        bias = self.user_bias_map.get(int(user_id), 0.0)
        return float(np.clip(base + 0.25 * bias, 1.0, 5.0))

    def recommend(self, user_id: int, seen_items: set, k: int = 10) -> list:
        recs = [iid for iid in self.ranked_items if iid not in seen_items]
        return recs[:k]


class PersonalizedHybridRecommender:
    """Personalized Matrix Factorization (TruncatedSVD) + Feature Store Genre Affinity Hybrid."""

    def __init__(self, n_components: int = 25, svd_weight: float = 0.80, random_state: int = 42):
        self.n_components = n_components
        self.svd_weight = svd_weight
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.user_factors = None
        self.item_factors = None
        self.user_means = None
        self.global_mean = 3.5
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

    def fit(self, matrix, mappings: dict):
        self.user_to_idx = {int(k): int(v) for k, v in mappings["user_to_idx"].items()}
        self.item_to_idx = {int(k): int(v) for k, v in mappings["item_to_idx"].items()}
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}
        self.global_mean = float(mappings["global_mean_rating"])

        dense = matrix.toarray().astype(np.float64)
        mask = dense > 0
        counts = mask.sum(axis=1)
        sums = dense.sum(axis=1)
        self.user_means = np.where(counts > 0, sums / np.maximum(counts, 1), self.global_mean)

        centered = np.where(mask, dense - self.user_means[:, None], 0.0)
        k_eff = min(self.n_components, min(centered.shape) - 1)
        self.svd = TruncatedSVD(n_components=k_eff, random_state=self.random_state)
        self.user_factors = self.svd.fit_transform(centered)
        self.item_factors = self.svd.components_
        return self

    def predict_rating(self, user_id: int, item_id: int, genre_affinity: float = 0.0) -> float:
        u_idx = self.user_to_idx.get(int(user_id))
        i_idx = self.item_to_idx.get(int(item_id))
        if u_idx is None or i_idx is None:
            return self.global_mean
        svd_pred = float(self.user_means[u_idx] + np.dot(self.user_factors[u_idx], self.item_factors[:, i_idx]))
        genre_boost = self.global_mean + (genre_affinity * 1.5)
        blended = self.svd_weight * svd_pred + (1.0 - self.svd_weight) * genre_boost
        return float(np.clip(blended, 1.0, 5.0))

    def recommend(self, user_id: int, seen_items: set, feature_store: RecommendationFeatureStore, k: int = 10) -> list:
        u_idx = self.user_to_idx.get(int(user_id))
        if u_idx is None:
            return []

        svd_scores = self.user_means[u_idx] + self.user_factors[u_idx] @ self.item_factors
        candidate_items = [iid for iid in self.item_to_idx.keys() if iid not in seen_items]
        online_feats = feature_store.get_online_features(user_id, candidate_items)

        aff_map = dict(zip(online_feats["item_id"].astype(int), online_feats["genre_affinity_score"].astype(float)))
        scored = []
        for iid in candidate_items:
            i_idx = self.item_to_idx[iid]
            aff = aff_map.get(iid, 0.0)
            score = self.svd_weight * float(svd_scores[i_idx]) + (1.0 - self.svd_weight) * (self.global_mean + aff * 1.5)
            scored.append((iid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in scored[:k]]


def run_training(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    features_dir = Path(cfg["paths"]["features_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    user_feats = pd.read_parquet(features_dir / "user_features.parquet")
    item_feats = pd.read_parquet(features_dir / "item_features.parquet")
    matrix = load_npz(processed_dir / "user_item_matrix.npz")
    with open(processed_dir / "matrix_mappings.json", "r") as f:
        mappings = json.load(f)

    mlflow.set_tracking_uri(cfg["training"]["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["training"]["mlflow_experiment"])

    # 1. Train & log Popularity Baseline
    with mlflow.start_run(run_name="Baseline_Popularity_Recommender") as run_base:
        pop_model = PopularityRecommender().fit(
            item_features=item_feats,
            user_features=user_feats,
            global_mean=mappings["global_mean_rating"],
        )
        pop_path = models_dir / "popularity_model.joblib"
        joblib.dump(pop_model, pop_path)

        mlflow.log_param("model_type", "BayesianPopularityBaseline")
        mlflow.log_param("bayesian_prior_weight", cfg["features"]["bayesian_prior_weight"])
        mlflow.log_artifact(str(pop_path))
        base_run_id = run_base.info.run_id

    # 2. Train & log Personalized Hybrid Recommender
    n_comp = int(cfg["training"]["svd_components"])
    svd_w = float(cfg["training"]["hybrid_svd_weight"])
    seed = int(cfg["project"]["random_seed"])

    with mlflow.start_run(run_name="Personalized_Hybrid_SVD_Recommender") as run_pers:
        hybrid_model = PersonalizedHybridRecommender(
            n_components=n_comp, svd_weight=svd_w, random_state=seed
        ).fit(matrix=matrix, mappings=mappings)
        hybrid_path = models_dir / "personalized_hybrid_model.joblib"
        joblib.dump(hybrid_model, hybrid_path)

        explained_var = float(hybrid_model.svd.explained_variance_ratio_.sum())
        mlflow.log_param("model_type", "TruncatedSVD_GenreHybrid")
        mlflow.log_param("n_components", n_comp)
        mlflow.log_param("svd_weight", svd_w)
        mlflow.log_param("random_seed", seed)
        mlflow.log_metric("svd_explained_variance_ratio", round(explained_var, 4))
        mlflow.log_artifact(str(hybrid_path))
        pers_run_id = run_pers.info.run_id

    summary = {
        "baseline_run_id": base_run_id,
        "personalized_run_id": pers_run_id,
        "svd_explained_variance": round(explained_var, 4),
    }
    logger.info(f"Model training complete: {summary}")
    return summary


if __name__ == "__main__":
    run_training()