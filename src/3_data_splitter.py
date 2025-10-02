from zenml.logger import get_logger
from typing import List, Tuple
import pandas as pd
from typing_extensions import Annotated
from zenml import step
from src.steps.model_train import DataSplitter
from src.exception.exception import CustomException
logger = get_logger(__name__)
import sys
@step
def splitting_data(
    df: pd.DataFrame
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        data_split = DataSplitter(df, features=df.drop('qty', axis = 1).columns, target="qty")
        X_train, X_test, y_train, y_test = data_split.split()
        logger.info("Data Split Successfully completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error("Error in Data splitting",e )
        raise CustomException(e, sys)
@step
def combine_data(
    X_train : pd.DataFrame,
    X_test : pd.DataFrame,
    y_train : pd.Series,
    y_test : pd.Series
) -> Tuple[
    Annotated[pd.DataFrame, "df_train"],
    Annotated[pd.DataFrame, "df_test"]
]:
    try:
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        df_train.rename(columns={"series": "qty"}, inplace=True)
        df_test.rename(columns={"series": "qty"}, inplace=True)

        logger.info("Data Combined successfully")
        print(df_train.columns)
        print(df_test.columns)

        return df_train, df_test
    
    except Exception as e:
        logger.info("Error in combining data",e )
        raise CustomException(e, sys)
