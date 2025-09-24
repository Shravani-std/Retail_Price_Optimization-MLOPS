# Shows how to build resilient pipelines that handle failures using retries and hooks

import random 
import time

from typing_extensions import Annotated
from zenml import pipeline, step
from zenml.config import StepRetryConfig
from zenml.logger import get_logger

try:
    from utils import log_dashboard_urls #type: ignore
except ImportError:
    log_dashboard_urls = lambda name: print(f" Pipeline '{name}' completed!")

logger = get_logger(__name__)

#Hook Function
def failure_hook(exc: BaseException):
    print(f"hook: step failed with {exc!r}")

@step(
    retry=StepRetryConfig(max_retries=3, delay=1, backoff=2),
    on_failure=failure_hook,
)
def flaky() -> Annotated[str, "result"]:
    if random.random() < 0.5:
        raise RuntimeError("Intermediate Error")
    time.sleep(0.5) 
    return "All good!"


@pipeline
def robust_pipeline():
    flaky()

if __name__=="__main__":
    run = robust_pipeline()
    step_run = run.steps['flaky']
    if step_run.status == "COMPLETED":
        msg = step_run.outputs["result"][0].load()
        logger.info(f"Final Result: {msg}")
    else:

        logger.info(f"Pipeline ended in state: {step_run.status}") 
    log_dashboard_urls("robust_pipeline")
