import logging
from typing import Tuple

import pandas as pd
from model.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)
from typing_extensions import Annotated
from zenml import step
from zenml.materializers.pandas_materializer import PandasMaterializer



@step(output_materializers={
        "x_train": PandasMaterializer,
        "x_test": PandasMaterializer,
        "y_train": PandasMaterializer,
        "y_test": PandasMaterializer,
    })
def clean_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    """
    if data is None or data.empty:
        raise ValueError("Received empty DataFrame in clean_data!")

    try:
        preprocess_strategy = DataPreprocessStrategy()
        # data= ingest_data()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        return x_train, x_test, y_train, y_test

    except Exception as e:
        logging.error(e)
        raise e


