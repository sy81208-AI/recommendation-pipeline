import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_CATALOG = [
    {
        "feature_name": "user_avg_rating",
        "entity": "User",
        "dtype": "float64",
        "description": "Mean explicit rating given by the user across historical interactions.",
        "source": "train_interactions",
        "usage": "Training & Inference",
    },
    {
        "feature_name": "user_activity_count",
        "entity": "User",
        "dtype": "int64",
        "description": "Total number of historical interactions recorded for the user.",
        "source": "train_interactions",
        "usage": "Training & Inference",
    },
    {
        "feature_name": "user_preferred_genre",
        "entity": "User",
        "dtype": "string",
        "description": "Most frequently interacted primary genre for the user.",
        "source": "train_interactions + clean_items",
        "usage": "Training & Inference",
    },
    {
        "feature_name": "days_since_last_interaction",
        "entity": "User",
        "dtype": "float64",
        "description": "Days elapsed between the user's most recent interaction and dataset cutoff.",
        "source": "train_interactions",
        "usage": "Training & Inference",
    },
    {
        "feature_name": "item_popularity_count",
        "entity": "Item",
        "dtype": "int64",
        "description": "Number of historical ratings received by the item.",
        "source": "item_popularity",
        "usage": "Training & Inference",
    },
    {
        "feature_name": "item_avg_rating",
        "entity": "Item",
        "dtype": "float64",
        "description": "Raw average explicit rating of the item.",
        "source": "item_popularity",
        "usage": "Training & Inference",
    },
    {
        "feature_name": "item_bayesian_rating",
        "entity": "Item",
        "dtype": "float64",
        "description": "Smoothed rating combining item average and global mean with prior weight m=10.",
        "source": "item_popularity",
        "usage": "Training & Inference",
    },
    {
        "feature_name": "primary_genre",
        "entity": "Item",
        "dtype": "string",
        "description": "Primary genre category of the item.",
        "source": "clean_items",
        "usage": "Training & Inference",
    },
    {
        "feature_name": "genre_affinity_score",
        "entity": "User-Item Interaction",
        "dtype": "float64",
        "description": "Share of the user's historical interactions matching the candidate item's primary genre.",
        "source": "Derived cross-feature",
        "usage": "Training & Inference",
    },
]


class RecommendationFeatureStore:
    """Unified offline/online feature store preventing training-serving skew."""

    def __init__(self, features_dir: str = "data/features"):
        self.features_dir = Path(features_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self._user_features = None
        self._item_features = None
        self._genre_affinity = None

    def materialize_features(self, processed_dir: str = "data/processed", global_mean: float = 3.5):
        p_dir = Path(processed_dir)
        users = pd.read_parquet(p_dir / "clean_users.parquet")
        items = pd.read_parquet(p_dir / "clean_items.parquet")
        train_df = pd.read_parquet(p_dir / "train_interactions.parquet")
        user_summary = pd.read_parquet(p_dir / "user_activity_summary.parquet")
        item_pop = pd.read_parquet(p_dir / "item_popularity.parquet")

        reference_ts = int(train_df["timestamp"].max())

        # Merge item genres onto interactions for genre preference & affinity
        inter_with_genre = train_df.merge(items[["item_id", "primary_genre"]], on="item_id", how="left")

        # User preferred genre
        genre_counts = (
            inter_with_genre.groupby(["user_id", "primary_genre"])
            .size()
            .reset_index(name="genre_interactions")
        )
        preferred_genre = (
            genre_counts.sort_values(["user_id", "genre_interactions"], ascending=[True, False])
            .drop_duplicates("user_id")[["user_id", "primary_genre"]]
            .rename(columns={"primary_genre": "user_preferred_genre"})
        )

        # User-Genre affinity matrix (long format)
        user_totals = genre_counts.groupby("user_id")["genre_interactions"].transform("sum")
        genre_counts["genre_affinity_score"] = (genre_counts["genre_interactions"] / user_totals).round(4)
        affinity_df = genre_counts[["user_id", "primary_genre", "genre_affinity_score"]]

        # Assemble User Feature Table
        user_feats = (
            users[["user_id", "age", "gender", "occupation"]]
            .merge(
                user_summary[["user_id", "interaction_count", "avg_rating", "last_ts"]],
                on="user_id",
                how="left",
            )
            .merge(preferred_genre, on="user_id", how="left")
        )
        user_feats["user_activity_count"] = user_feats["interaction_count"].fillna(0).astype(int)
        user_feats["user_avg_rating"] = user_feats["avg_rating"].fillna(global_mean).round(4)
        user_feats["user_preferred_genre"] = user_feats["user_preferred_genre"].fillna("Drama")
        user_feats["days_since_last_interaction"] = (
            ((reference_ts - user_feats["last_ts"].fillna(reference_ts)) / 86400.0)
            .clip(lower=0.0)
            .round(2)
        )
        user_feats = user_feats[
            [
                "user_id",
                "age",
                "gender",
                "occupation",
                "user_activity_count",
                "user_avg_rating",
                "user_preferred_genre",
                "days_since_last_interaction",
            ]
        ]

        # Assemble Item Feature Table
        item_feats = items[["item_id", "title", "primary_genre", "genres", "release_year"]].merge(
            item_pop[["item_id", "rating_count", "avg_rating", "bayesian_rating", "popularity_rank"]],
            on="item_id",
            how="left",
        )
        item_feats["item_popularity_count"] = item_feats["rating_count"].fillna(0).astype(int)
        item_feats["item_avg_rating"] = item_feats["avg_rating"].fillna(global_mean).round(4)
        item_feats["item_bayesian_rating"] = item_feats["bayesian_rating"].fillna(global_mean).round(4)
        item_feats["popularity_rank"] = item_feats["popularity_rank"].fillna(len(item_feats) + 1).astype(int)
        item_feats = item_feats[
            [
                "item_id",
                "title",
                "primary_genre",
                "genres",
                "release_year",
                "item_popularity_count",
                "item_avg_rating",
                "item_bayesian_rating",
                "popularity_rank",
            ]
        ]

        # Assemble Training Feature View
        training_features = self._join_features(train_df[["user_id", "item_id", "rating", "timestamp"]], user_feats, item_feats, affinity_df)

        user_feats.to_parquet(self.features_dir / "user_features.parquet", index=False)
        item_feats.to_parquet(self.features_dir / "item_features.parquet", index=False)
        affinity_df.to_parquet(self.features_dir / "user_genre_affinity.parquet", index=False)
        training_features.to_parquet(self.features_dir / "training_features.parquet", index=False)

        self._user_features = user_feats
        self._item_features = item_feats
        self._genre_affinity = affinity_df
        return user_feats, item_feats, affinity_df

    def _join_features(self, pairs_df: pd.DataFrame, user_feats: pd.DataFrame, item_feats: pd.DataFrame, affinity_df: pd.DataFrame) -> pd.DataFrame:
        df = pairs_df.merge(user_feats, on="user_id", how="left").merge(item_feats, on="item_id", how="left")
        df = df.merge(affinity_df, on=["user_id", "primary_genre"], how="left")
        df["genre_affinity_score"] = df["genre_affinity_score"].fillna(0.0)
        return df

    def load_store(self):
        if self._user_features is None:
            self._user_features = pd.read_parquet(self.features_dir / "user_features.parquet")
            self._item_features = pd.read_parquet(self.features_dir / "item_features.parquet")
            self._genre_affinity = pd.read_parquet(self.features_dir / "user_genre_affinity.parquet")

    def get_online_features(self, user_id: int, candidate_item_ids: list) -> pd.DataFrame:
        """Serves identical feature vectors at inference/prediction time."""
        self.load_store()
        pairs = pd.DataFrame({"user_id": [user_id] * len(candidate_item_ids), "item_id": candidate_item_ids})
        return self._join_features(pairs, self._user_features, self._item_features, self._genre_affinity)


def export_feature_catalog(reports_dir: str = "reports"):
    r_dir = Path(reports_dir)
    r_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Feature Catalog",
        "",
        "| Feature Name | Entity | Data Type | Description | Source | Usage |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for row in FEATURE_CATALOG:
        lines.append(
            f"| `{row['feature_name']}` | {row['entity']} | `{row['dtype']}` | "
            f"{row['description']} | {row['source']} | {row['usage']} |"
        )
    (r_dir / "feature_catalog.md").write_text("\n".join(lines), encoding="utf-8")


def run_feature_engineering(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    store = RecommendationFeatureStore(cfg["paths"]["features_dir"])
    user_feats, item_feats, affinity_df = store.materialize_features(cfg["paths"]["processed_dir"])
    export_feature_catalog(cfg["paths"]["reports_dir"])

    summary = {
        "user_feature_rows": len(user_feats),
        "item_feature_rows": len(item_feats),
        "user_genre_affinity_rows": len(affinity_df),
        "catalog_features_documented": len(FEATURE_CATALOG),
    }
    logger.info(f"Feature engineering complete: {summary}")
    return summary


if __name__ == "__main__":
    run_feature_engineering()