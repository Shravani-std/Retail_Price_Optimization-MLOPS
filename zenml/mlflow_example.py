import mlflow
from zenml import step, pipeline
from typing import Tuple
from typing_extensions import Annotated
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load data
@step
def load_data() -> Annotated[Tuple, "X, y"]:
    data = load_diabetes()
    return data.data, data.target

# Step 2: Train model
@step
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    with mlflow.start_run(nested=True):
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("train_score", model.score(X, y))
    return model

# Step 3: Evaluate model
@step
def evaluate_model(model, X, y):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    with mlflow.start_run(nested=True):
        mlflow.log_metric("mse", mse)
    return mse

# Pipeline
@pipeline
def training_pipeline(loader, trainer, evaluator):
    X, y = loader()           # now safely unpack
    model = trainer(X, y)
    mse = evaluator(model, X, y)
    return mse

# Run pipeline
if __name__ == "__main__":
    loader_step = load_data
    trainer_step = train_model
    evaluator_step = evaluate_model

    pipe = training_pipeline(
        loader=loader_step,
        trainer=trainer_step,
        evaluator=evaluator_step
    )
    pipe.run()
