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