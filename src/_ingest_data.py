from src.exception.exception import CustomException

import pandas as pd
from typing_extensions import Annotated
from zenml import step
from src.steps.data_loader import DataLoader
import sys
from zenml.logger import get_logger

logger = get_logger(__name__)

@step(enable_cache=False)
def data_ingestion(table_name: str, for_predict: bool = False) -> pd.DataFrame:
    try:
        data_loader = DataLoader('/media/shrav/New Volume/AI/MLOPS/Retail_Price_Optimization-MLOPS-/data/retail_price.csv')
        data_loader.load_data(table_name)
        df = data_loader.get_data()
        if for_predict:
            df.drop(columns=["qty"], inplace=True)
        print(df.head())
       
        logger.info("Data loaded successfully")
        return df
    except Exception as e:

        logger.error("Error in data loading")
        raise CustomException(e, sys)
    
