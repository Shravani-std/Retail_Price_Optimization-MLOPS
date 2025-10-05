import sys
from typing import Tuple
import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger
from src.steps.data_preprocessing import CategoricalEncoder, OutlierHandler
from src.steps.feature_eng import DateFeatureEngineer
from src.exception.exception import CustomException
logger = get_logger(__name__)

@step
def categoricsl_encode(df: pd.DataFrame) -> pd.DataFrame:
    try:
        print(df.head())
        encoder = CategoricalEncoder(method="onehot")
        df = encoder.fit_transform(df, columns=["product_id", "product_category_name"])
        
        logger.info("Categorical encoding applied to categoricsl columns")

        #Handling Outliers
        outlier_handler = OutlierHandler(multiplier=1.5)
        df_transformed = outlier_handler.fit_transform(df, columns=["total_price", "freight_price", "unit_price"])
        
        logger.info("Outlier handling is completed.") 

        
        logger.info("Data processed successfully")

        return df_transformed
    except Exception as e:
    
        logger.error("Error has occured during data processing")       
        raise CustomException(e, sys)
    
@step
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    try:
        data_engineering = DateFeatureEngineer(date_format="%Y-%m-%d")
        df_transf = data_engineering.fit_transform(df, ['month_year'])

        logger.info("Feature engineering applied sucessfully")

        df_transf.drop(columns=[col for col in ["month_year"] if col in df_transf], inplace=True)


        return df_transf
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")

        raise CustomException(e, sys)