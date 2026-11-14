```markdown
# End-to-End Data Management Pipeline for a Recommendation System

A modular, reproducible data management and machine learning pipeline built for **Data Management for Machine Learning (Assignment 2)**. The system ingests user, item, and interaction data, validates data quality with quarantine logging, performs time-aware preparation and sparse matrix transformations, manages reusable offline/online features, trains and tracks baseline and personalized models in MLflow, and orchestrates execution via Prefect.

## Project Structure
```text
recommendation-pipeline/
├── configs/config.yaml                # Centralized pipeline parameters
├── dags/recommendation_flow.py        # Prefect orchestration workflow & schedule
├── data/
│   ├── raw/                           # Ingested raw CSV files
│   ├── processed/                     # Validated & prepared Parquet + sparse NPZ
│   ├── features/                      # Materialized offline/online feature tables
│   └── errors/                        # Quarantined invalid records with error tags
├── models/                            # Serialized model artifacts (.joblib)
├── reports/                           # Validation report, feature catalog, eval table, diagrams
├── src/
│   ├── ingestion/ingest.py
│   ├── validation/validate.py
│   ├── preparation/prepare.py
│   ├── transformation/transform.py
│   ├── features/feature_store.py
│   ├── training/train.py
│   └── evaluation/evaluate.py
├── requirements.txt
└── README.md