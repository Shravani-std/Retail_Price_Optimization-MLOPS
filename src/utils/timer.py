import time 
from logger.logger import logging
import time
from src.logger.logger import logging

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

