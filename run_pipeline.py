from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)


if __name__ == "__main__":
    # training = continuous_deployment_pipeline()
    training = train_pipeline()

    print(f"Pipeline triggered successfully! Run ID: {training.id}")

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )


