import io
import logging
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


def _generate_synthetic_fallback(seed: int = 42):
    rng = np.random.default_rng(seed)
    n_users, n_items, n_interactions = 300, 150, 12000

    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "age": rng.integers(18, 65, size=n_users),
        "gender": rng.choice(["M", "F"], size=n_users),
        "occupation": rng.choice(["engineer", "educator", "student", "artist", "scientist"], size=n_users),
        "zip_code": [f"{rng.integers(10000, 99999)}" for _ in range(n_users)],
    })

    primary_genres = rng.choice(GENRES[1:], size=n_items)
    items = pd.DataFrame({
        "item_id": np.arange(1, n_items + 1),
        "title": [f"Movie Title {i} ({rng.integers(1985, 2025)})" for i in range(1, n_items + 1)],
        "release_date": [f"01-Jan-{rng.integers(1985, 2025)}" for _ in range(n_items)],
        "primary_genre": primary_genres,
        "genres": primary_genres,
    })

    base_ts = 874724710
    interactions = pd.DataFrame({
        "user_id": rng.integers(1, n_users + 1, size=n_interactions),
        "item_id": rng.integers(1, n_items + 1, size=n_interactions),
        "rating": rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], p=[0.06, 0.11, 0.27, 0.34, 0.22], size=n_interactions),
        "timestamp": base_ts + rng.integers(0, 18000000, size=n_interactions),
    })
    return users, items, interactions


def _inject_quality_audit_records(interactions: pd.DataFrame) -> pd.DataFrame:
    """Appends controlled dirty records to verify automated validation & quarantine."""
    valid_sample = interactions.iloc[:3].copy()
    dirty_rows = pd.DataFrame([
        # 1. Duplicate records
        valid_sample.iloc[0].to_dict(),
        valid_sample.iloc[1].to_dict(),
        # 2. Invalid ratings (out of [1, 5] range)
        {"user_id": 1, "item_id": 10, "rating": 7.5, "timestamp": 881250949},
        {"user_id": 2, "item_id": 20, "rating": -1.0, "timestamp": 881250949},
        {"user_id": 3, "item_id": 30, "rating": 0.0, "timestamp": 881250949},
        # 3. Missing values
        {"user_id": 4, "item_id": 40, "rating": None, "timestamp": 881250949},
        {"user_id": None, "item_id": 50, "rating": 4.0, "timestamp": 881250949},
        # 4. Invalid data types
        {"user_id": "bad_id", "item_id": 15, "rating": "five", "timestamp": "not_a_ts"},
        # 5. Invalid timestamps (pre-1995 and far future)
        {"user_id": 5, "item_id": 12, "rating": 4.0, "timestamp": 100000},
        {"user_id": 6, "item_id": 14, "rating": 3.0, "timestamp": 2524608000},
        # 6. Referential integrity violations (non-existent users and items)
        {"user_id": 99999, "item_id": 10, "rating": 5.0, "timestamp": 881250949},
        {"user_id": 88888, "item_id": 12, "rating": 4.0, "timestamp": 881250949},
        {"user_id": 10, "item_id": 77777, "rating": 3.0, "timestamp": 881250949},
        {"user_id": 11, "item_id": 66666, "rating": 2.0, "timestamp": 881250949},
        {"user_id": 99999, "item_id": 77777, "rating": 1.0, "timestamp": 881250949},
    ])
    return pd.concat([interactions, dirty_rows], ignore_index=True)


def run_ingestion(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = cfg["ingestion"]["dataset_url"]

    try:
        logger.info(f"Downloading MovieLens 100K dataset from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            zip_bytes = io.BytesIO(resp.read())

        with zipfile.ZipFile(zip_bytes) as zf:
            users = pd.read_csv(
                zf.open("ml-100k/u.user"),
                sep="|",
                names=["user_id", "age", "gender", "occupation", "zip_code"],
                encoding="latin-1",
            )

            item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
            raw_items = pd.read_csv(
                zf.open("ml-100k/u.item"),
                sep="|",
                names=item_cols,
                encoding="latin-1",
            )
            genre_matrix = raw_items[GENRES].values
            primary_genres = [
                GENRES[row.argmax()] if row.sum() > 0 else "unknown" for row in genre_matrix
            ]
            all_genres = [
                "|".join([GENRES[i] for i, val in enumerate(row) if val == 1]) or "unknown"
                for row in genre_matrix
            ]
            items = pd.DataFrame({
                "item_id": raw_items["item_id"],
                "title": raw_items["title"],
                "release_date": raw_items["release_date"],
                "primary_genre": primary_genres,
                "genres": all_genres,
            })

            interactions = pd.read_csv(
                zf.open("ml-100k/u.data"),
                sep="\t",
                names=["user_id", "item_id", "rating", "timestamp"],
            )
        source_used = "MovieLens-100K"
    except Exception as exc:
        logger.warning(f"Remote download unavailable ({exc}). Using deterministic synthetic fallback.")
        users, items, interactions = _generate_synthetic_fallback(cfg["project"]["random_seed"])
        source_used = "Synthetic-Fallback"

    if cfg["ingestion"].get("inject_audit_anomalies", True):
        interactions = _inject_quality_audit_records(interactions)

    users.to_csv(raw_dir / "users.csv", index=False)
    items.to_csv(raw_dir / "items.csv", index=False)
    interactions.to_csv(raw_dir / "interactions.csv", index=False)

    summary = {
        "source": source_used,
        "raw_users": len(users),
        "raw_items": len(items),
        "raw_interactions": len(interactions),
    }
    logger.info(f"Ingestion complete: {summary}")
    return summary


if __name__ == "__main__":
    run_ingestion()