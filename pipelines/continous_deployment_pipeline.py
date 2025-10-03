from zenml import pipeline
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from src._data_splitter import splitting_data
from src._deployment_trigger_step import deployment_trigger
from src._evaluator import evaluate
from src._ingest_data import data_ingestion
from src._data_process import categoricsl_encode, feature_engineer

# from steps.refine_model import remove_insignificant_vars
from src._train_model import re_train, sklearn_train

docker_settings = DockerSettings(required_integrations=[MLFLOW]) 


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.9,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = data_ingestion("retail_prices")
    df_processed = categoricsl_encode(df)
    df_transformed = feature_engineer(df_processed)  
    X_train, X_test, y_train, y_test = splitting_data(df_transformed) 
    model, predictors = sklearn_train(X_train, y_train) 
    # predictors = remove_insignificant_vars(model, df_transformed, alpha=0.05) 
    # rmse = evaluate(model, df_transformed)
    # model1, df_with_significant_vars = re_train(X_train, y_train, predictors)   
    
    # rmse1 = evaluate(model1, df_with_significant_vars)

    # deployment_decision = deployment_trigger(
    #     accuracy=rmse1, min_accuracy=min_accuracy
    # )
    mlflow_model_deployer_step(
        model=model,
        # deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )