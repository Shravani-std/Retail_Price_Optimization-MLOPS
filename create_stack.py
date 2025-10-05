from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.artifact_stores import LocalArtifactStore
from zenml.orchestrators import LocalOrchestrator

client = Client()

# 1️⃣ Create artifact store
artifact_store = LocalArtifactStore(
    path="/media/shrav/New Volume/AI/MLOPS/Retail_Price_Optimization-MLOPS-/zenml_artifacts/mlruns"
)
artifact_store = client.active_stack.register_artifact_store(artifact_store)

# 2️⃣ Create orchestrator
orchestrator = LocalOrchestrator()
orchestrator = client.active_stack.register_orchestrator(orchestrator)

# 3️⃣ Create MLflow experiment tracker
experiment_tracker = MLFlowExperimentTracker()
experiment_tracker = client.active_stack.register_experiment_tracker(experiment_tracker)

# 4️⃣ Create stack
stack = client.create_stack(
    name="retail_stack",
    orchestrator=orchestrator,
    artifact_store=artifact_store,
    experiment_tracker=experiment_tracker,
)

# 5️⃣ Activate stack
client.activate_stack(stack.name)

print("Stack created and activated successfully!")
