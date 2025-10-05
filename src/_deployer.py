from zenml.integrations.bentoml.steps import bentoml_model_deployer_step
from constants import MODEL_NAME

bentoml_model_deployer = bentoml_model_deployer_step.with_options(
    parameters=dict(
        model_name=MODEL_NAME,
        port=3001,
        production=False,
        timeout=1000,
    )
)