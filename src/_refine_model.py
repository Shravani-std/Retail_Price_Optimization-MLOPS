from typing import List
import statsmodels.api as sm # type: ignore
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor # type: ignore
from zenml.logger import get_logger
import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper # type: ignore
from typing_extensions import Annotated
from zenml import step
from materializer.custom_materializer import StatsModelMaterializer # type: ignore
from src.steps.model_train import LinearRegressionModel, ModelRefinement
from src.exception.exception import CustomException
import sys
logger = get_logger(__name__)

@step 
def remove_insignificant_vars(
    model: RegressionResultsWrapper,
    df: pd.DataFrame,
    alpha: float = 0.05,
) -> Annotated[List[str], "significant_preditors"]:
    try:
        print(type(model))
        print(model.summary())
        refinement = ModelRefinement(model,df)
        preditors = refinement.remove_insignificant_vars(alpha=alpha)
        logger.info("Model Refined successfully")
        return preditors
    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)