from zenml import pipeline
from zenml.logger import get_logger
logger = get_logger(__name__)
from zenml.client import Client
print(Client().active_stack.experiment_tracker.get_tracking_uri())
from zenml.config import DockerSettings
from zenml.integrations.constants import BENTOML
from src._ingest_data import data_ingestion
from src._bentoml_builder import bento_builder
from src._data_splitter import combine_data, splitting_data
from src._deployer import bentoml_model_deployer
from src._evaluator import evaluate
from src._data_process import categoricsl_encode, feature_engineer
from src._refine_model import remove_insignificant_vars
from src._train_model import re_train, sklearn_train

from src._deployment_trigger_step import deployment_trigger

docker_settings = DockerSettings(required_integrations=[BENTOML, DEEPCHECKS])




@pipeline(enable_cache=False, settings={"docker": docker_settings})

def training_retail():
    df = data_ingestion("retail_prices")
    df_processed = categoricsl_encode(df)
    df_tranformed = feature_engineer(df_processed)

    X_train, X_test, y_train, y_test = splitting_data(df_tranformed)

    model, predictors = sklearn_train(X_train, y_train)

    rmse = 0.95

    decision = deployment_trigger(accuracy=rmse, min_accuracy=0.80)
    bento = bento_builder(model=model)
    bentoml_model_deployer(bento=bento, deploy_decision = decision)
