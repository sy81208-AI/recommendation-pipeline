import logging
from pathlib import Path
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_preparation(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    users = pd.read_parquet(processed_dir / "validated_users.parquet")
    items = pd.read_parquet(processed_dir / "validated_items.parquet")
    interactions = pd.read_parquet(processed_dir / "validated_interactions.parquet")

    # 1. Standardize user & item fields and handle missing values
    users["age"] = users["age"].fillna(users["age"].median()).astype(int)
    users["gender"] = users["gender"].str.strip().str.upper()
    users["occupation"] = users["occupation"].str.strip().str.lower()
    users["zip_code"] = users["zip_code"].fillna("00000").astype(str).str.strip()

    items["title"] = items["title"].str.strip()
    items["primary_genre"] = items["primary_genre"].str.strip()
    items["release_year"] = (
        pd.to_datetime(items["release_date"], errors="coerce")
        .dt.year.fillna(1995)
        .astype(int)
    )

    # 2. Standardize timestamps & deduplicate repeated user-item interactions (keep latest)
    interactions["event_datetime"] = pd.to_datetime(interactions["timestamp"], unit="s", utc=True)
    interactions = (
        interactions.sort_values("timestamp")
        .drop_duplicates(subset=["user_id", "item_id"], keep="last")
        .reset_index(drop=True)
    )

    # 3. Time-based chronological split per user (80% earliest -> train, 20% latest -> test)
    test_ratio = float(cfg["preparation"]["test_ratio"])
    min_count = int(cfg["preparation"]["min_user_interactions_for_split"])

    train_parts, test_parts = [], []
    for _, user_df in interactions.groupby("user_id"):
        user_df = user_df.sort_values("timestamp")
        n = len(user_df)
        if n >= min_count:
            split_idx = int(n * (1.0 - test_ratio))
            train_parts.append(user_df.iloc[:split_idx])
            test_parts.append(user_df.iloc[split_idx:])
        else:
            train_parts.append(user_df)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    users.to_parquet(processed_dir / "clean_users.parquet", index=False)
    items.to_parquet(processed_dir / "clean_items.parquet", index=False)
    train_df.to_parquet(processed_dir / "train_interactions.parquet", index=False)
    test_df.to_parquet(processed_dir / "test_interactions.parquet", index=False)

    summary = {
        "clean_users": len(users),
        "clean_items": len(items),
        "train_interactions": len(train_df),
        "test_interactions": len(test_df),
        "split_strategy": f"Per-user chronological ({int((1-test_ratio)*100)}/{int(test_ratio*100)})",
    }
    logger.info(f"Preparation complete: {summary}")
    return summary


if __name__ == "__main__":
    run_preparation()