# from abc import ABC, abstractmethod
from typing import List
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
# from src.data_loader import DataLoader
from src.exception.exception import CustomException
from src.logger.logger import logging
from zenml.logger import get_logger
from zenml import step

logger = get_logger(__name__)


class CategoricalEncoder:

    def __init__(self, method="onehot", categories='auto'):
        self.method = method
        self.categories = categories
        self.encoders = {}

    def fit(self, df, columns):
        try:
            logging.info(f"Fitting CategoricalEncoder with method={self.method} on columns={columns}")
            for col in columns:
                if self.method == "onehot":
                    self.encoders[col] = OneHotEncoder(sparse_output=False, categories=self.categories)
                elif self.method == "ordinal":
                    self.encoders[col] = OrdinalEncoder(categories=self.categories)
                else:
                    raise ValueError(f"Unsupported encoding method: {self.method}")
                self.encoders[col].fit(df[[col]])
            logging.info("CategoricalEncoder fitting completed successfully.")
        except Exception as e:
            logging.error("Error while fitting CategoricalEncoder.")
            raise CustomException(e, sys)

    def transform(self, df, columns):
        try:
            logging.info(f"Transforming dataframe using method={self.method} on columns={columns}")
            df_encoded = df.copy()
            for col in columns:
                transformed = self.encoders[col].transform(df[[col]])
                if self.method == "onehot":
                    transformed = pd.DataFrame(
                        transformed,
                        columns=self.encoders[col].get_feature_names_out([col]),
                        index=df.index
                    )
                    df_encoded = pd.concat([df_encoded.drop(columns=[col]), transformed], axis=1)
                else:
                    df_encoded[col] = transformed
            logging.info("CategoricalEncoder transformation completed successfully.")
            return df_encoded
        except Exception as e:
            logging.error("Error while transforming dataframe in CategoricalEncoder.")
            raise CustomException(e, sys)

    def fit_transform(self, df, columns):
        try:
            logging.info(f"Running fit_transform for CategoricalEncoder on columns={columns}")
            self.fit(df, columns)
            return self.transform(df, columns)
        except Exception as e:
            logging.error("Error in fit_transform of CategoricalEncoder.")
            raise CustomException(e, sys)

@step
def encode_categorical(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    try:
        encoder = CategoricalEncoder(method="onehot")
        df_encoded = encoder.fit_transform(df, categorical_columns)
        logger.info(f"Categorical encoding completed with shape {df.shape}")
        return df_encoded
    
    except Exception as e:
        logger.error("Error while encoding catgorical columns.")
        raise CustomException(e, sys)
    

class OutlierHandler:
    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier
        self.medians = {}
        self.iqr_bounds = {}
        self.outliers = pd.DataFrame()  # fixed typo (- changed to =)

    def fit(self, df: pd.DataFrame, columns: List[str]):
        try:
            logging.info(f"Fitting OutlierHandler on columns={columns} with multiplier={self.multiplier}")
            for col in columns:
                self.medians[col] = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                self.iqr_bounds[col] = (q1 - self.multiplier * iqr, q3 + self.multiplier * iqr)
            logging.info("OutlierHandler fitting completed successfully.")
        except Exception as e:
            logging.error("Error while fitting OutlierHandler.")
            raise CustomException(e, sys)

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        try:
            logging.info(f"Transforming dataframe with OutlierHandler on columns={columns}")
            for col in columns:
                outliers = df[(df[col] < self.iqr_bounds[col][0]) | (df[col] > self.iqr_bounds[col][1])]
                self.outliers = pd.concat([self.outliers, outliers])
                df[col] = np.where(
                    (df[col] < self.iqr_bounds[col][0]) | (df[col] > self.iqr_bounds[col][1]),
                    self.medians[col],
                    df[col]
                )
            logging.info("OutlierHandler transformation completed successfully.")
            return df
        except Exception as e:
            logging.error("Error while transforming dataframe in OutlierHandler.")
            raise CustomException(e, sys)

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        try:
            logging.info(f"Running fit_transform for OutlierHandler on columns={columns}")
            self.fit(df, columns)
            return self.transform(df, columns)
        except Exception as e:
            logging.error("Error in fit_transform of OutlierHandler.")
            raise CustomException(e, sys)
@step
def handle_outliers(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    try:
        outlier_handler = OutlierHandler(multiplier=1.5)
        df_transformed = outlier_handler.fit_transform(df, numeric_columns)
        logger.info(
            f"Outlier handling completed. Transformed shape: {df_transformed.shape}, "
            f"Outliers detected: {outlier_handler.outliers.shape[0]}"
        )

        return df_transformed
    except Exception as e:
        logger.error("Error while handling outliers.")
        raise CustomException(e, sys)
    
@step
def save_data(df:pd.DataFrame, output_path : str)-> None:
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved successfullu at {output_path} with shape {df.shape}")
    except Exception as e:
        logger.error("Error while saving data.")
        raise CustomException(e, sys)

    








# if __name__ == "__main__":
#     try:
#         logging.info("Starting ETL process with CategoricalEncoder and OutlierHandler...")

#         # --- Step 1: Load data from CSV ---
#         input_file = "/media/shrav/New Volume/AI/MLOPS/Retail_Price_Optimization-MLOPS-/data/retail_price.csv"   
#         df = pd.read_csv(input_file)
#         logging.info(f"Data loaded successfully from {input_file} with shape {df.shape}")

#         # --- Step 2: Encode categorical columns ---
#         categorical_columns = ["product_id", "product_category_name"] 
#         encoder = CategoricalEncoder(method="onehot")
#         df = encoder.fit_transform(df, columns=categorical_columns)

#         encoded_file = "/media/shrav/New Volume/AI/MLOPS/Retail_Price_Optimization-MLOPS-/data/retail_prices_encoded.csv"
#         df.to_csv(encoded_file, index=False)
#         logging.info(f"Categorical encoding completed. Encoded data saved at {encoded_file} with shape {df.shape}")

#         # --- Step 3: Load encoded data (if needed for next stage) ---
#         encoded_date_file = "/media/shrav/New Volume/AI/MLOPS/Retail_Price_Optimization-MLOPS-/data/retail_prices_encoded.csv" 
#         df = pd.read_csv(encoded_date_file)
#         logging.info(f"Encoded DataFrame loaded for outlier handling with shape {df.shape}")

#         # --- Step 4: Handle outliers ---
#         numeric_columns = ["total_price", "freight_price", "unit_price"] 
#         outlier_handler = OutlierHandler(multiplier=1.5)
#         df_transformed = outlier_handler.fit_transform(df, columns=numeric_columns)

#         logging.info(f"Outlier handling completed. Transformed DataFrame shape: {df_transformed.shape}")
#         logging.info(f"Outliers detected: {outlier_handler.outliers.shape[0]} rows")

#         # --- Step 5: Save transformed data ---
#         transformed_file = "/media/shrav/New Volume/AI/MLOPS/Retail_Price_Optimization-MLOPS-/data/retail_prices_transformed.csv"
#         df_transformed.to_csv(transformed_file, index=False)
#         logging.info(f"Final transformed DataFrame saved at {transformed_file}")

#     except Exception as e:
#         logging.error("Error occurred during ETL process.")
#         raise CustomException(e, sys)
