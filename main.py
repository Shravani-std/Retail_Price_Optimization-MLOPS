from zenml.client import Client
active_stack = Client().active_stack
experiment_tracker = active_stack.experiment_tracker

print(experiment_tracker.get_tracking_uri())
