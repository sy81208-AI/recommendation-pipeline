import json
import logging
from pathlib import Path
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_transformation(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    users = pd.read_parquet(processed_dir / "clean_users.parquet")
    items = pd.read_parquet(processed_dir / "clean_items.parquet")
    train_df = pd.read_parquet(processed_dir / "train_interactions.parquet")

    # 1. User Activity Summary
    user_summary = (
        train_df.groupby("user_id")
        .agg(
            interaction_count=("rating", "count"),
            avg_rating=("rating", "mean"),
            std_rating=("rating", "std"),
            min_rating=("rating", "min"),
            max_rating=("rating", "max"),
            first_ts=("timestamp", "min"),
            last_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    user_summary["std_rating"] = user_summary["std_rating"].fillna(0.0)
    user_summary["active_span_days"] = (
        (user_summary["last_ts"] - user_summary["first_ts"]) / 86400.0
    ).round(2)

    # 2. Item Popularity Summary
    global_mean = float(train_df["rating"].mean())
    m = float(cfg["features"]["bayesian_prior_weight"])

    item_pop = (
        train_df.groupby("item_id")
        .agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean"),
            std_rating=("rating", "std"),
        )
        .reset_index()
    )
    item_pop["std_rating"] = item_pop["std_rating"].fillna(0.0)
    item_pop["bayesian_rating"] = (
        (item_pop["rating_count"] / (item_pop["rating_count"] + m)) * item_pop["avg_rating"]
        + (m / (item_pop["rating_count"] + m)) * global_mean
    )
    item_pop = item_pop.sort_values(
        by=["rating_count", "bayesian_rating"], ascending=False
    ).reset_index(drop=True)
    item_pop["popularity_rank"] = item_pop.index + 1

    # 3. Sparse User-Item Interaction Matrix (CSR)
    all_user_ids = sorted(users["user_id"].unique().tolist())
    all_item_ids = sorted(items["item_id"].unique().tolist())
    user_to_idx = {int(uid): idx for idx, uid in enumerate(all_user_ids)}
    item_to_idx = {int(iid): idx for idx, iid in enumerate(all_item_ids)}

    rows = [user_to_idx[int(u)] for u in train_df["user_id"]]
    cols = [item_to_idx[int(i)] for i in train_df["item_id"]]
    vals = train_df["rating"].astype(float).values

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(all_user_ids), len(all_item_ids)))
    save_npz(processed_dir / "user_item_matrix.npz", matrix)

    with open(processed_dir / "matrix_mappings.json", "w") as f:
        json.dump(
            {
                "user_to_idx": {str(k): v for k, v in user_to_idx.items()},
                "item_to_idx": {str(k): v for k, v in item_to_idx.items()},
                "global_mean_rating": global_mean,
            },
            f,
            indent=2,
        )

    user_summary.to_parquet(processed_dir / "user_activity_summary.parquet", index=False)
    item_pop.to_parquet(processed_dir / "item_popularity.parquet", index=False)

    sparsity = 1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
    summary = {
        "user_summaries": len(user_summary),
        "item_popularity_rows": len(item_pop),
        "matrix_shape": list(matrix.shape),
        "matrix_sparsity_pct": round(sparsity * 100, 2),
    }
    logger.info(f"Transformation complete: {summary}")
    return summary


if __name__ == "__main__":
    run_transformation()