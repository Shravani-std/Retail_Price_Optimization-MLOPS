from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys


class FeatureEngineering(ABC):
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        pass


class DateFeatureEngineer(FeatureEngineering):
    def __init__(self, date_format: str = "%m-%d-%Y"):
        """Constructor"""
        self.date_format = date_format

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        try:
            logging.info(
                f"Running DateFeatureEngineer with columns={columns} and format={self.date_format}"
            )

            for col in columns:
                if col not in df.columns:   # ✅ check in dataframe, not in the list itself
                    raise ValueError(f"Column '{col}' not found in DataFrame")

                df = self._split_date(df, col)

            logging.info("Date Feature engineering completed successfully.")
            return df

        except Exception as e:
            logging.error("Error in DateFeatureEngineer fit_transform")
            raise CustomException(e, sys)

    def _split_date(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Splits a date column into month and year"""
        try:
            logging.info(f"Splitting column '{column}' into month and year")

            df[column] = pd.to_datetime(df[column], format=self.date_format, errors="coerce")

            if df[column].isnull().any():
                logging.warning(
                    f"Some rows in '{column}' could not be parsed with format {self.date_format}"
                )

            df[f"{column}_month"] = df[column].dt.month
            df[f"{column}_year"] = df[column].dt.year

            logging.info(
                f"Successfully split column '{column}' into '{column}_month' and '{column}_year'"
            )
            return df

        except Exception as e:
            logging.error(f"Error while splitting date column: {column}")
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     try:
#         input_file = "/media/shrav/New Volume/AI/MLOPS/Retail_Price_Optimization-MLOPS-/data/retail_prices_transformed.csv"
#         output_file = "/media/shrav/New Volume/AI/MLOPS/Retail_Price_Optimization-MLOPS-/data/retail_prices_transformed_date.csv"

#         df = pd.read_csv(input_file)
#         logging.info(f"Data loaded successfully with shape {df.shape}")

#         data_eng = DateFeatureEngineer(date_format="%Y-%m-%d")
#         df_transformed = data_eng.fit_transform(df, ["month_year"])   # ✅ pass as list

#         df_transformed.to_csv(output_file, index=False)
#         logging.info(f"Transformed data saved to {output_file}")

#     except Exception as e:
#         raise CustomException(e, sys)
