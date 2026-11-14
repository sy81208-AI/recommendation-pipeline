import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when invoked from dags/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prefect import flow, task, get_run_logger
from src.ingestion.ingest import run_ingestion
from src.validation.validate import run_validation
from src.preparation.prepare import run_preparation
from src.transformation.transform import run_transformation
from src.features.feature_store import run_feature_engineering
from src.training.train import run_training
from src.evaluation.evaluate import run_evaluation


@task(name="1_Data_Ingestion", retries=2, retry_delay_seconds=5)
def ingest_task(config_path: str):
    get_run_logger().info("Starting raw data ingestion...")
    return run_ingestion(config_path)


@task(name="2_Data_Validation")
def validate_task(config_path: str, ingest_meta: dict):
    get_run_logger().info(f"Validating dataset from source: {ingest_meta['source']}")
    return run_validation(config_path)


@task(name="3_Data_Preparation")
def prepare_task(config_path: str, val_meta: dict):
    get_run_logger().info(f"Preparing {val_meta['valid_interactions']} validated records...")
    return run_preparation(config_path)


@task(name="4_Data_Transformation")
def transform_task(config_path: str, prep_meta: dict):
    get_run_logger().info(f"Building derived tables & sparse matrix ({prep_meta['train_interactions']} train rows)...")
    return run_transformation(config_path)


@task(name="5_Feature_Management")
def feature_task(config_path: str, trans_meta: dict):
    get_run_logger().info(f"Materializing offline/online features (sparsity: {trans_meta['matrix_sparsity_pct']}%)...")
    return run_feature_engineering(config_path)


@task(name="6_Model_Training")
def train_task(config_path: str, feat_meta: dict):
    get_run_logger().info(f"Training models using {feat_meta['catalog_features_documented']} cataloged features...")
    return run_training(config_path)


@task(name="7_Model_Evaluation")
def evaluate_task(config_path: str, train_meta: dict):
    get_run_logger().info("Evaluating models and logging metrics to MLflow...")
    return run_evaluation(config_path, run_ids=train_meta)


@flow(name="End-to-End-Recommendation-Data-Pipeline", log_prints=True)
def recommendation_pipeline_flow(config_path: str = "configs/config.yaml"):
    ingest_res = ingest_task(config_path)
    val_res = validate_task(config_path, ingest_res)
    prep_res = prepare_task(config_path, val_res)
    trans_res = transform_task(config_path, prep_res)
    feat_res = feature_task(config_path, trans_res)
    train_res = train_task(config_path, feat_res)
    eval_res = evaluate_task(config_path, train_res)
    print("Pipeline completed successfully!")
    return eval_res


if __name__ == "__main__":
    if "--serve" in sys.argv:
        # Runs on a daily 02:00 AM cron schedule when served
        recommendation_pipeline_flow.serve(
            name="daily-recommendation-pipeline",
            cron="0 2 * * *",
        )
    else:
        recommendation_pipeline_flow()