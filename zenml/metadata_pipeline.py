from typing_extensions import Annotated
from zenml import log_metadata, pipeline, step
from zenml.logger import get_logger

try:
    from utils import log_dashboard_urls #type: ignore
except ImportError:
    log_dashboard_urls = lambda name: print(f" Pipeline '{name}' completed")

logger = get_logger(__name__)


@step
def compute_accuracy() -> Annotated[float, "acuuracy_metric"]:
    logger.info("Computing model Accuracy")
    acc = 0.95
    logger.info(f"Accuracy Computed: {acc}")
    log_metadata({"accuracy" : acc})
    logger.info("Accuracy metadata logged to Zenml")
    return acc

@pipeline
def metadata_pipeline():
    compute_accuracy()

if __name__=="__main__":
    logger.info("Starting metadata logging Pipeline")
    metadata_pipeline()
    logger.info("Pipeline Completed")

    log_dashboard_urls("metadata_p[ipeline]")