# from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from src.exception.exception import CustomException
# from src.logger.logger import logging
import sys
from zenml.logger import get_logger
from zenml import step
logger = get_logger(__name__)



class DateFeatureEngineer():
    def __init__(self, date_format: str = "%m-%d-%Y"):
        """Constructor"""
        self.date_format = date_format

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        try:
            logger.info(
                f"Running DateFeatureEngineer with columns={columns} and format={self.date_format}"
            )

            for col in columns:
                if col not in df.columns:  
                    raise ValueError(f"Column '{col}' not found in DataFrame")

                df = self._split_date(df, col)

            logger.info("Date Feature engineering completed successfully.")
            return df

        except Exception as e:
            logger.error("Error in DateFeatureEngineer fit_transform")
            raise CustomException(e, sys)

    def _split_date(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Splits a date column into month and year"""
        try:
            logger.info(f"Splitting column '{column}' into month and year")

            df[column] = pd.to_datetime(df[column], format=self.date_format, errors="coerce")

            if df[column].isnull().any():
                logger.warning(
                    f"Some rows in '{column}' could not be parsed with format {self.date_format}"
                )

            df[f"{column}_month"] = df[column].dt.month
            df[f"{column}_year"] = df[column].dt.year

            logger.info(
                f"Successfully split column '{column}' into '{column}_month' and '{column}_year'"
            )
            return df

        except Exception as e:
            logger.error(f"Error while splitting date column: {column}")
            raise CustomException(e, sys)

@step 
def date_feature_engineering_step(df: pd.DataFrame, date_columns: List[str], date_format:str = "%Y-%m-%d") -> pd.DataFrame:
    try:
        engineer = DateFeatureEngineer(date_format=date_format)
        df_transformed = engineer.fit_transform(df, date_format)
        logger.info(f"Date feature enginerring completed. Transformed df shape {df_transformed.shape}")
        return df_transformed
    except Exception as e:
        logger.error("Error in date_feature_enginerring _step")
        raise CustomException(e, sys)
    


# if __name__ == "__main__":
#     try:
#         input_file = "/media/shrav/New Volume/AI/MLOPS/Retail_Price_Optimization-MLOPS-/data/retail_prices_transformed.csv"
#         output_file = "/media/shrav/New Volume/AI/MLOPS/Retail_Price_Optimization-MLOPS-/data/retail_prices_transformed_date.csv"

#         df = pd.read_csv(input_file)
#         logger.info(f"Data loaded successfully with shape {df.shape}")

#         data_eng = DateFeatureEngineer(date_format="%Y-%m-%d")
#         df_transformed = data_eng.fit_transform(df, ["month_year"])   # ✅ pass as list

#         df_transformed.to_csv(output_file, index=False)
#         logger.info(f"Transformed data saved to {output_file}")

#     except Exception as e:
#         raise CustomException(e, sys)
