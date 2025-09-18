import sys
from src.logger import logging



def error_message_detail(error, error_detail: sys):
    """
    Build a detailed error message including file name, line number, and the original error message.
    Must be called inside an except block
    """
    _, _, exc_tb = error_detail.exc_info()

    while exc_tb.tb_next:
        exc_tb = exc_tb.tb_next

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error)
    )
    return error_message


class CustomException(Exception):   # inherit from Exception
    def __init__(self, error_message, error_detail: sys):
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        super().__init__(self.error_message)

        logging.error(self.error_message)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divided by zero")
        raise CustomException(str(e), sys)
