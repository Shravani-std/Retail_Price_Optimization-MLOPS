from zenml import __version__ as zenml_version
from zenml.integrations.bentoml.steps import bento_builder_step

bento_builder = bento_builder_step.with_options(
    parameters=dict(
        model_name=MODEL_NAME,
        model_type="sklearn",
        servise="service.py:svc",
        labels={
            "framwork":"sklearn",
            "dataset":"retail",
            "zenml_version":"",
        },
        exclude=["data"],
        python={
            "packages": ["zenml","scikit-learn"],
        },
    )
)