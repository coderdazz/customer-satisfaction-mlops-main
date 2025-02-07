import logging
import pickle
from typing import Annotated

import mlflow
import pandas as pd
from zenml.materializers import CloudpickleMaterializer

from model.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from materializer.custom_materializer import ModelMaterializer
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker


# @step(experiment_tracker=experiment_tracker.name)
@step(enable_cache=False, output_materializers={"output": CloudpickleMaterializer})
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str = "linear_regression",
    fine_tuning: bool = False
) -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        model = None
        tuner = None

        if model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            model = LightGBMModel()
        elif model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif model_name == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBoostModel()
        elif model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            raise ValueError("Model name not supported")

        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        if fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            trained_model = model.train(x_train, y_train)


        # ✅ Log model to MLflow
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(trained_model,
                                     artifact_path="model",
                                     signature=mlflow.models.infer_signature(x_train, trained_model.predict(x_train)),
                                     )
            mlflow.log_params({"model": "model"})
            mlflow.log_params(trained_model.get_params())
            mlflow.log_metric("train_score", trained_model.score(x_train, y_train))
            mlflow.log_metric("test_score", trained_model.score(x_test, y_test))

        print(f"✅ DEBUG: Model logged to MLflow. Run ID: {run.info.run_id}")

        return trained_model

    except Exception as e:
        logging.error(e)
        raise e