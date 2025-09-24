from typing import Annotated

import pandas as pd
from zenml import ArtifactConfig, Tag, add_tags, pipeline, step
from zenml.logger import get_logger

try:
    from utils import log_dashboard_urls # type: ignore
except ImportError:
    log_dashboard_urls = lambda name: print(f" Pipeline '{name} completd!")

logger = get_logger(__name__)



@step
def create_raw_data() -> Annotated[
    pd.DataFrame, ArtifactConfig(name="raw_data", tags=["raw", "input"])
]:
    """Create Raw data with Artifact level tags."""
    data = pd.DataFrame(
        {
            "feature_1":[1,2,3,4,5],
            "feature_2":[10,20,30,40,50],
             "target":[0,1,0,1,0],
        }
    )
    logger.info(f"Created raw data with shape: {data.shape}")
    return data

@step 
def process_data(
    raw_data:pd.DataFrame,

) -> Annotated[
    pd.DataFrame,ArtifactConfig(name="processed_data", tags=["processed"])
]:
    """Process data and add dynamic tags."""
    # Simple processing: Normalize Features
    processed =  raw_data.copy()
    processed["feature_1"] = (
        processed["feature_1"] / processed["feature_1"].max()
    )

    processed["feature_2"] = (
        processed["feature_2"] / processed["feature_2"].max()
    )

    #Add tags
    add_tags(tags=["normalized", "ready_for_training"], infer_artifact=True)

    logger.info("Processed data with Normalization")
    return processed


@pipeline(tags=["Tutorial Example", Tag(name="Experiment", cascade=True)])
def run_pipeline():
    data = create_raw_data()
    processed_data = process_data(data)
    return processed_data

if __name__=="__main__":
    run_pipeline()
    log_dashboard_urls("run_pipeline")
    logger.info("Run to see how tags works")