My Github Url: https://github.com/sy81208-AI/recommendation-pipeline


reports\validation_report.json
{
  "total_raw_interactions": 100015,
  "valid_interactions": 100000,
  "quarantined_invalid_interactions": 15,
  "validation_pass_rate_pct": 99.99,
  "error_breakdown": {
    "DUPLICATE_RECORD": 2,
    "INVALID_RATING": 3,
    "MISSING_VALUE": 2,
    "INVALID_DATA_TYPE": 3,
    "INVALID_TIMESTAMP": 2,
    "ORPHAN_USER_ID": 3,
    "ORPHAN_ITEM_ID": 3
  }
}


# Model Evaluation Comparison (K = 10)

| Metric | Popularity Baseline | Personalized Hybrid (SVD + Genre) |
| :--- | :---: | :---: |
| **Precision@10** | 0.0787 | 0.0935 |
| **Recall@10** | 0.0805 | 0.0904 |
| **NDCG@10** | 0.0995 | 0.1200 |
| **Coverage@10** | 0.0434 | 0.2634 |
| **Diversity@10** | 0.7481 | 0.6047 |
| **RMSE** | 1.0457 | 1.1241 |
| **MAE** | 0.8402 | 0.8933 |


# Feature Catalog

| Feature Name | Entity | Data Type | Description | Source | Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `user_avg_rating` | User | `float64` | Mean explicit rating given by the user across historical interactions. | train_interactions | Training & Inference |
| `user_activity_count` | User | `int64` | Total number of historical interactions recorded for the user. | train_interactions | Training & Inference |
| `user_preferred_genre` | User | `string` | Most frequently interacted primary genre for the user. | train_interactions +clean_items | Training & Inference |
| `days_since_last_interaction` | User | `float64` | Days elapsed between the user's most recent interaction and dataset cutoff. | train_interactions | Training & Inference |
| `item_popularity_count` | Item | `int64` | Number of historical ratings received by the item. | item_popularity | Training & Inference |
| `item_avg_rating` | Item | `float64` | Raw average explicit rating of the item. | item_popularity | Training & Inference|
| `item_bayesian_rating` | Item | `float64` | Smoothed rating combining item average and global mean with prior weight m=10. | item_popularity | Training & Inference |
| `primary_genre` | Item | `string` | Primary genre category of the item. | clean_items | Training & Inference |
| `genre_affinity_score` | User-Item Interaction | `float64` | Share of the user's historical interactions matching the candidate item's primary genre. | Derived cross-feature | Training & Inference |

# End-to-End Recommendation Data Management Pipeline Architecture

```mermaid
flowchart LR
    A[("1. Data Ingestion<br/>MovieLens 100K + QA Audit<br/>(data/raw/*.csv)")] --> B{"2. Data Validation<br/>Pandera + Integrity Rules"}
    B -->|Invalid Rows| E[("Quarantine Store<br/>data/errors/invalid_interactions.csv<br/>reports/validation_report.json")]
    B -->|Clean Rows| C["3. Data Preparation<br/>Dedup, Standardization,<br/>Chronological 80/20 Split"]
    C --> D["4. Data Transformation<br/>User Summary, Item Popularity,<br/>CSR Sparse Interaction Matrix"]
    D --> F[("5. Feature Management<br/>RecommendationFeatureStore<br/>Parquet + Feature Catalog")]
    F --> G["6. Model Training<br/>Popularity Baseline &<br/>SVD + Genre Hybrid"]
    G <--> M[("MLflow Tracking<br/>Params, Metrics, Models")]
    G --> H["7. Model Evaluation<br/>Precision@10, Recall@10, NDCG@10,<br/>Coverage, Diversity, RMSE, MAE"]
    H --> R[("Reports & Artifacts<br/>reports/evaluation_results.md")]

    O[["Orchestration Layer: Prefect DAG (dags/recommendation_flow.py)"]] -.-> A & B & C & D & F & G & H

    