from typing import List, Tuple
import mlflow
import mlflow.sklearn
import mlflow.statsmodels
import pandas as pd
import statsmodels.api as sm  # type: ignore
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import RegressionResultsWrapper  # type: ignore
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger
from materializer.custom_materializer import (  # type: ignore
    ListMaterializer,
    SKLearnModelMaterializer,
    StatsModelMaterializer,
)
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from sklearn.impute import SimpleImputer
import sys
from src.steps.model_train import LinearRegressionModel
from src.exception.exception import CustomException

logger = get_logger(__name__)

# Ensure active stack has MLflow
active_stack = Client().active_stack
if active_stack is None:
    raise RuntimeError("No active stack found. Please set a ZenML stack first.")

experiment_tracker = active_stack.experiment_tracker
if not experiment_tracker or not isinstance(experiment_tracker, MLFlowExperimentTracker):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for this example to work."
    )

def preprocess_X(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess DataFrame for LinearRegression:
    - Keep only numeric columns
    - Drop fully empty numeric columns
    - Impute remaining NaNs with mean
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include="number").columns
    # Drop fully empty columns
    valid_cols = [col for col in numeric_cols if df_copy[col].notna().any()]
    df_copy = df_copy[valid_cols]
    # Impute NaNs
    if len(df_copy.columns) > 0:
        imputer = SimpleImputer(strategy="mean")
        df_copy[df_copy.columns] = imputer.fit_transform(df_copy)
    return df_copy

@step(
    experiment_tracker="mlflow_tracker_retail",
    settings={"experiment_tracker.mlflow": {"experiment_name": "test_name"}},
    enable_cache=False,
    output_materializers=[SKLearnModelMaterializer, ListMaterializer]
)
def sklearn_train(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.Series, "y_train"],
) -> Tuple[
    Annotated[LinearRegression, "model"],
    Annotated[List[str], "predictors"],
]:
    try:
        X_train_clean = preprocess_X(X_train)

        mlflow.end_run()
        with mlflow.start_run() as run:
            mlflow.sklearn.autolog()
            model = LinearRegression()
            model.fit(X_train_clean, y_train)
            predictors = X_train_clean.columns.tolist()
            logger.info(f"Predictors used: {predictors}")
            return model, predictors
    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)

@step(
    experiment_tracker="mlflow_tracker_retail",
    settings={"experiment_tracker.mlflow": {"experiment_name": "test_name"}},
    output_materializers=[StatsModelMaterializer, ListMaterializer]
)
def re_train(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.Series, "y_train"],
    predictors: list
) -> Tuple[
    Annotated[RegressionResultsWrapper, "model"],
    Annotated[pd.DataFrame, "df_with_significant_vars"],
]:
    """Trains a linear regression model and outputs the summary."""
    try:
        # Select only predictor columns and preprocess
        X_train_selected = preprocess_X(X_train[predictors])

        model = LinearRegressionModel(X_train_selected, y_train)
        mlflow.statsmodels.autolog()
        trained_model = model.train()

        # Combine X and y into one DataFrame
        df_with_significant_vars = pd.concat(
            [X_train_selected.reset_index(drop=True), y_train.reset_index(drop=True)],
            axis=1
        )
        df_with_significant_vars.rename(columns={"series": "qty"}, inplace=True)

        logger.info("Model trained successfully")
        return trained_model, df_with_significant_vars
    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)


# if __name__ == "__main__":
#     import numpy as np

#     # Example dummy data
#     X_train = pd.DataFrame({
#         "feature1": np.random.rand(10),
#         "feature2": np.random.rand(10),
#         "feature3": np.random.rand(10),
#         "empty_col": [None]*10,  # fully empty column to test
#     })
#     y_train = pd.Series(np.random.rand(10))

#     # Run sklearn_train step
#     try:
#         model, predictors = sklearn_train(X_train, y_train)
#         print("sklearn_train completed")
#         print("Predictors:", predictors)
#         print("Predictions:", model.predict(preprocess_X(X_train)))
#     except Exception as e:
#         print(f"sklearn_train failed: {e}")

#     # Run re_train step
#     try:
#         trained_model, df_with_significant_vars = re_train(X_train, y_train, predictors)
#         print("re_train completed")
#         print(df_with_significant_vars.head())
#     except Exception as e:
#         print(f"re_train failed: {e}")
