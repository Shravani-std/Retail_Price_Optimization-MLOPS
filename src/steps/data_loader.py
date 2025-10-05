
import pandas as pd
from src.exception.exception import CustomException
from src.logger.logger import logging
import sys
import time


class DataLoader:
    """
    Loading Data from Excel file
    """
    def __init__(self, file_path: str):
        
        self.file_path = file_path
        self.data = None

    def timed(func):
        """Decorator for execution time logging"""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logging.info(f"'{func.__name__}' executed in {end - start:.4f}s")
            print(f"'{func.__name__}' executed in {end - start:.4f}s")
            return result
        return wrapper
   
    @timed
    def load_data(self, sheet_name: str = 0) -> pd.DataFrame:
        """
        Load Data from the Excel
        """
        try:
            self.data = pd.read_csv('/media/shrav/New Volume/AI/MLOPS/Retail_Price_Optimization-MLOPS-/data/retail_price.csv')
            logging.info(f"Data loaded Successfully from {self.file_path}")
            return self.data
        except Exception as e:
            logging.error(f"Failed to load file {self.file_path}")
            raise CustomException(e, sys)

    def get_data(self) -> pd.DataFrame:
        """
        Returns the Loaded data : pd.DataFrame
        """
        if self.data is not None:
            logging.info("Returning loaded data.")
            return self.data
        else:
            logging.warning("Tried to access data before loading")
            raise CustomException("Data not loaded yet. Please call 'load_data' first.", sys)
        


# if __name__=="__main__":

#     data_loader = DataLoader("data/retail_price.csv")
#     df = data_loader.load_data()
#     logging.info("Data loading is completed!")
#     print(df.head())