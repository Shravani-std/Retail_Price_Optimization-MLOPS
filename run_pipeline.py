from zenml.logger import get_logger
from zenml.client import Client
from pipelines.training_pipeline import training_retail
from src._ingest_data import data_ingestion
from src._data_process import categoricsl_encode # fix typo

logger = get_logger(__name__)

if __name__ == "__main__":
    # Print MLflow tracking URI
    print(Client().active_stack.experiment_tracker.get_tracking_uri())

    # Run pipeline
    pipeline_instance = training_retail()
    pipeline_instance.run()
