from typing import List
import sys
import pandas as pd
import statsmodels.api as sm # type: ignore
from statsmodel.regression.linear_model import RegressionResultsWrapper # type: ignore
from typing_extensions import Annotated
from zenml.logger import get_logger
from src.exception.exception import CustomException
from src.steps.model_train import ModelRefinement
import mlflow 
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml import step
logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )

@step(experiment_tracker=experiment_tracker.name)
def evaluate(
    model: RegressionResultsWrapper, 
    df: pd.DataFrame,
)-> Annotated[float, "rmse"]:
    try:
        refinement = ModelRefinement(model.df)
        rmse = refinement.validate()
        logger.info(f"RMSE: {rmse}")
        logger.info("Model Evaluated Successfully")
        mlflow.log_metric("emse", rmse)

        return rmse
    except Exception as e:
        logger.error(e)
        raise CustomException(e,sys)