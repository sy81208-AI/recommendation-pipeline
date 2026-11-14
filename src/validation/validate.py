import json
import logging
from pathlib import Path
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_validation(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    errors_dir = Path(cfg["paths"]["errors_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])

    for d in (processed_dir, errors_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    users = pd.read_csv(raw_dir / "users.csv")
    items = pd.read_csv(raw_dir / "items.csv")
    interactions = pd.read_csv(raw_dir / "interactions.csv", dtype=str)

    # 1. Validate Users & Items via Pandera
    user_schema = DataFrameSchema({
        "user_id": Column(int, checks=Check.gt(0), unique=True, nullable=False),
        "age": Column(float, checks=Check.in_range(5, 110), nullable=True, coerce=True),
        "gender": Column(str, checks=Check.isin(["M", "F", "O"]), nullable=False),
        "occupation": Column(str, nullable=False),
        "zip_code": Column(str, nullable=True, coerce=True),
    })
    item_schema = DataFrameSchema({
        "item_id": Column(int, checks=Check.gt(0), unique=True, nullable=False),
        "title": Column(str, nullable=False),
        "release_date": Column(str, nullable=True),
        "primary_genre": Column(str, nullable=False),
        "genres": Column(str, nullable=False),
    })

    valid_users = user_schema.validate(users)
    valid_items = item_schema.validate(items)

    valid_user_ids = set(valid_users["user_id"].tolist())
    valid_item_ids = set(valid_items["item_id"].tolist())

    min_rating = float(cfg["validation"]["min_rating"])
    max_rating = float(cfg["validation"]["max_rating"])
    min_ts = pd.Timestamp(cfg["validation"]["min_timestamp"], tz="UTC")
    max_ts = pd.Timestamp(cfg["validation"]["max_timestamp"], tz="UTC")

    # 2. Row-by-row interaction validation with explicit error tagging
    error_reasons = []
    seen_keys = set()

    parsed_user_ids = pd.to_numeric(interactions["user_id"], errors="coerce")
    parsed_item_ids = pd.to_numeric(interactions["item_id"], errors="coerce")
    parsed_ratings = pd.to_numeric(interactions["rating"], errors="coerce")
    parsed_ts_num = pd.to_numeric(interactions["timestamp"], errors="coerce")

    for idx in range(len(interactions)):
        reasons = []
        raw_row = interactions.iloc[idx]

        # Missing values check
        if raw_row.isna().any() or (raw_row.astype(str).str.strip() == "").any():
            reasons.append("MISSING_VALUE")

        uid = parsed_user_ids.iloc[idx]
        iid = parsed_item_ids.iloc[idx]
        rat = parsed_ratings.iloc[idx]
        ts_val = parsed_ts_num.iloc[idx]

        # Data type check
        if pd.isna(uid) or pd.isna(iid) or pd.isna(rat) or pd.isna(ts_val):
            reasons.append("INVALID_DATA_TYPE")
        else:
            uid_int, iid_int = int(uid), int(iid)
            # Rating range check
            if not (min_rating <= float(rat) <= max_rating):
                reasons.append(f"INVALID_RATING({rat})")

            # Timestamp bounds check
            try:
                dt = pd.to_datetime(int(ts_val), unit="s", utc=True)
                if not (min_ts <= dt <= max_ts):
                    reasons.append(f"INVALID_TIMESTAMP({ts_val})")
            except Exception:
                reasons.append(f"UNPARSEABLE_TIMESTAMP({ts_val})")

            # Referential integrity check
            if uid_int not in valid_user_ids:
                reasons.append(f"ORPHAN_USER_ID({uid_int})")
            if iid_int not in valid_item_ids:
                reasons.append(f"ORPHAN_ITEM_ID({iid_int})")

            # Duplicate check
            dedup_key = (uid_int, iid_int, int(ts_val))
            if dedup_key in seen_keys:
                reasons.append("DUPLICATE_RECORD")
            else:
                seen_keys.add(dedup_key)

        error_reasons.append(";".join(reasons) if reasons else "")

    interactions["error_reason"] = error_reasons
    invalid_mask = interactions["error_reason"] != ""

    invalid_df = interactions[invalid_mask].copy()
    clean_df = interactions[~invalid_mask].drop(columns=["error_reason"]).copy()

    clean_df["user_id"] = clean_df["user_id"].astype(int)
    clean_df["item_id"] = clean_df["item_id"].astype(int)
    clean_df["rating"] = clean_df["rating"].astype(float)
    clean_df["timestamp"] = clean_df["timestamp"].astype(int)

    # Final Pandera contract enforcement on clean interactions
    interaction_schema = DataFrameSchema({
        "user_id": Column(int, checks=Check.isin(valid_user_ids)),
        "item_id": Column(int, checks=Check.isin(valid_item_ids)),
        "rating": Column(float, checks=Check.in_range(min_rating, max_rating)),
        "timestamp": Column(int, checks=Check.gt(0)),
    })
    clean_df = interaction_schema.validate(clean_df)

    # Save quarantined errors & validated parquet datasets
    invalid_df.to_csv(errors_dir / "invalid_interactions.csv", index=False)
    valid_users.to_parquet(processed_dir / "validated_users.parquet", index=False)
    valid_items.to_parquet(processed_dir / "validated_items.parquet", index=False)
    clean_df.to_parquet(processed_dir / "validated_interactions.parquet", index=False)

    # Breakdown of error categories
    breakdown = {}
    for entry in invalid_df["error_reason"]:
        for tag in entry.split(";"):
            category = tag.split("(")[0]
            breakdown[category] = breakdown.get(category, 0) + 1

    report = {
        "total_raw_interactions": int(len(interactions)),
        "valid_interactions": int(len(clean_df)),
        "quarantined_invalid_interactions": int(len(invalid_df)),
        "validation_pass_rate_pct": round(100.0 * len(clean_df) / max(len(interactions), 1), 2),
        "error_breakdown": breakdown,
    }
    with open(reports_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Validation summary: {report}")
    return report


if __name__ == "__main__":
    run_validation()