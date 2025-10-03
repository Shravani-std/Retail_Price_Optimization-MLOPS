# from pipelines.training_pipeline import training_pipeline
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)

from pipelines.training_pipeline import training_pipeline
from src._ingest_data import data_ingestion
from src._data_process import categoricsl_encode

if __name__ == "__main__":
    training_pipeline()