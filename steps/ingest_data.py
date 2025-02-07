import logging
import pandas as pd
from zenml import step
from zenml.materializers.pandas_materializer import PandasMaterializer
from typing_extensions import Annotated

@step(output_materializers={"data": PandasMaterializer})
def ingest_data() -> Annotated[pd.DataFrame,"data"]:
    """
    Args:
        None

    Returns:
        df: pd.DataFrame
    """
    try:
        # ingest_data = IngestData()
        df = pd.read_csv("./data/olist_customers_dataset.csv")

        # print(f"🟢 DEBUG: ingest_data() output type: {type(df)}")
        # print(f"🟢 DEBUG: ingest_data() output shape: {df.shape if isinstance(df, pd.DataFrame) else 'Not a DataFrame'}")

        return df.copy()

    except Exception as e:
        logging.error(e)
        raise e

