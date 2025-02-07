import os
import pickle
from typing import Any, Type, Union

import numpy as np
import pandas as pd
# from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor, Booster
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from sklearn.base import RegressorMixin

DEFAULT_FILENAME = "CustomerSatisfactionEnvironment"


class ModelMaterializer(BaseMaterializer):
    """Materializer to handle Regressor mixing models."""

    ASSOCIATED_TYPES = (str, RegressorMixin)
    def handle_input(self, data_type: Type[Any]) -> Any:
        """Loads an object from the artifact storage."""
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            obj = pickle.load(fid)

        if not isinstance(obj, self.ASSOCIATED_TYPES):
            raise TypeError(f"Loaded object is not of expected types: {self.ASSOCIATED_TYPES}")

        return obj

    def handle_return(self, obj: Any) -> None:
        """Saves an object (model or dataset) to the artifact storage."""
        if not isinstance(obj, self.ASSOCIATED_TYPES):
            raise TypeError(f"Cannot save object of type {type(obj)} using cs_materializer.")

        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(obj, fid)

#
# class cs_materializer(BaseMaterializer):
#     """
#     Custom materializer for the Customer Satisfaction Project
#     """
#
#     ASSOCIATED_TYPES = (
#         str,
#         np.ndarray,
#         pd.Series,
#         pd.DataFrame,
#         CatBoostRegressor,
#         RandomForestRegressor,
#         LGBMRegressor,
#         XGBRegressor,
#     )
#
#     def handle_input(
#         self, data_type: Type[Any]
#     ) -> Union[
#         str,
#         np.ndarray,
#         pd.Series,
#         pd.DataFrame,
#         CatBoostRegressor,
#         RandomForestRegressor,
#         LGBMRegressor,
#         XGBRegressor,
#     ]:
#         """
#         It loads the model from the artifact and returns it.
#
#         Args:
#             data_type: The type of the model to be loaded
#         """
#         filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
#         with fileio.open(filepath, "rb") as fid:
#             obj = pickle.load(fid)
#         return obj
#
#     def handle_return(
#         self,
#         obj: Union[
#             str,
#             np.ndarray,
#             pd.Series,
#             pd.DataFrame,
#             CatBoostRegressor,
#             RandomForestRegressor,
#             LGBMRegressor,
#             XGBRegressor,
#         ],
#     ) -> None:
#         """
#         It saves the model to the artifact store.
#
#         Args:
#             model: The model to be saved
#         """
#
#         filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
#         with fileio.open(filepath, "wb") as fid:
#             pickle.dump(obj, fid)
