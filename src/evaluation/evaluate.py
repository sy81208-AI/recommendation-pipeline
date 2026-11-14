import json
import logging
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import yaml

from src.features.feature_store import RecommendationFeatureStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = 0.0
    for idx, iid in enumerate(recommended[:k]):
        if iid in relevant:
            dcg += 1.0 / np.log2(idx + 2)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg


def _intra_list_diversity(recommended: list, genre_vectors: dict) -> float:
    vecs = [genre_vectors[iid] for iid in recommended if iid in genre_vectors]
    if len(vecs) < 2:
        return 0.0
    sim_matrix = cosine_similarity(np.vstack(vecs))
    upper_tri = sim_matrix[np.triu_indices(len(vecs), k=1)]
    return float(1.0 - np.mean(upper_tri))


def run_evaluation(config_path: str = "configs/config.yaml", run_ids: dict = None) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    features_dir = Path(cfg["paths"]["features_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(processed_dir / "train_interactions.parquet")
    test_df = pd.read_parquet(processed_dir / "test_interactions.parquet")
    item_feats = pd.read_parquet(features_dir / "item_features.parquet")
    affinity_df = pd.read_parquet(features_dir / "user_genre_affinity.parquet")

    pop_model = joblib.load(models_dir / "popularity_model.joblib")
    hybrid_model = joblib.load(models_dir / "personalized_hybrid_model.joblib")
    feature_store = RecommendationFeatureStore(str(features_dir))
    feature_store.load_store()

    k = int(cfg["evaluation"]["top_k"])
    rel_thresh = float(cfg["evaluation"]["relevance_threshold"])

    # Precompute seen training items per user
    user_seen = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    # Precompute item multi-hot genre vectors for Intra-List Diversity
    genre_dummies = item_feats["genres"].str.get_dummies(sep="|")
    genre_vectors = {
        int(iid): genre_dummies.iloc[idx].values.astype(float)
        for idx, iid in enumerate(item_feats["item_id"])
    }
    total_catalog_size = len(item_feats)

    # Prepare explicit rating test pairs with genre affinity
    test_enriched = test_df.merge(item_feats[["item_id", "primary_genre"]], on="item_id", how="left").merge(
        affinity_df, on=["user_id", "primary_genre"], how="left"
    )
    test_enriched["genre_affinity_score"] = test_enriched["genre_affinity_score"].fillna(0.0)

    # Ground-truth relevant test items per user
    relevant_by_user = (
        test_df[test_df["rating"] >= rel_thresh]
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )
    eval_users = sorted(relevant_by_user.keys())

    results = {}
    for model_name, model in [
        ("Popularity Baseline", pop_model),
        ("Personalized Hybrid (SVD + Genre)", hybrid_model),
    ]:
        # 1. Explicit Rating Metrics (RMSE & MAE)
        y_true = test_enriched["rating"].values
        if model_name == "Popularity Baseline":
            y_pred = [
                model.predict_rating(int(r.user_id), int(r.item_id))
                for r in test_enriched.itertuples()
            ]
        else:
            y_pred = [
                model.predict_rating(int(r.user_id), int(r.item_id), float(r.genre_affinity_score))
                for r in test_enriched.itertuples()
            ]
        rmse = float(root_mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))

        # 2. Top-K Ranking, Coverage & Diversity Metrics
        precisions, recalls, ndcgs, diversities = [], [], [], []
        all_recommended_items = set()

        for uid in eval_users:
            seen = user_seen.get(uid, set())
            rel_set = relevant_by_user[uid]
            if model_name == "Popularity Baseline":
                recs = model.recommend(uid, seen_items=seen, k=k)
            else:
                recs = model.recommend(uid, seen_items=seen, feature_store=feature_store, k=k)

            hits = len(set(recs) & rel_set)
            precisions.append(hits / k)
            recalls.append(hits / len(rel_set))
            ndcgs.append(_ndcg_at_k(recs, rel_set, k))
            diversities.append(_intra_list_diversity(recs, genre_vectors))
            all_recommended_items.update(recs)

        coverage = len(all_recommended_items) / max(total_catalog_size, 1)

        metrics = {
            f"Precision@{k}": round(float(np.mean(precisions)), 4),
            f"Recall@{k}": round(float(np.mean(recalls)), 4),
            f"NDCG@{k}": round(float(np.mean(ndcgs)), 4),
            f"Coverage@{k}": round(float(coverage), 4),
            f"Diversity@{k}": round(float(np.mean(diversities)), 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
        }
        results[model_name] = metrics

    # Log metrics to MLflow
    mlflow.set_tracking_uri(cfg["training"]["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["training"]["mlflow_experiment"])
    if run_ids:
        for key, m_name in [
            ("baseline_run_id", "Popularity Baseline"),
            ("personalized_run_id", "Personalized Hybrid (SVD + Genre)"),
        ]:
            if key in run_ids:
                with mlflow.start_run(run_id=run_ids[key]):
                    for metric_k, metric_v in results[m_name].items():
                        mlflow.log_metric(metric_k.replace("@", "_at_"), metric_v)

    # Save Comparison Markdown Table & JSON
    with open(reports_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    base_m = results["Popularity Baseline"]
    pers_m = results["Personalized Hybrid (SVD + Genre)"]
    md_lines = [
        f"# Model Evaluation Comparison (K = {k})",
        "",
        f"| Metric | Popularity Baseline | Personalized Hybrid (SVD + Genre) |",
        "| :--- | :---: | :---: |",
    ]
    for metric_name in base_m.keys():
        md_lines.append(f"| **{metric_name}** | {base_m[metric_name]:.4f} | {pers_m[metric_name]:.4f} |")

    report_md = "\n".join(md_lines)
    (reports_dir / "evaluation_results.md").write_text(report_md, encoding="utf-8")
    logger.info("\n" + report_md)
    return results


if __name__ == "__main__":
    run_evaluation()