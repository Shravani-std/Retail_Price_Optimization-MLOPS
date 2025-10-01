from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step
from zenml.logger import get_logger
from src.exception.exception import CustomException
import sys

import statsmodels.api as sm
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

logger  = get_logger(__name__)

@step
def split_date(df:  pd.DataFrame, features: List[str], target: str, test_size : float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    try:

        logger.info(f"Splitting data into train.test with test_size = {test_size}")
        X = df [features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        logger.info(f"Train_shape: {X_train.shape}, Test shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error("Error in splitting data")
        raise CustomException(e, sys)


@step
def linear_regression_step(X_train: pd.DataFrame, y_train: pd.Series) -> sm.OLS:
    """
    Trains a linear regression model using statsmodels.
    """
    X_train_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit()
    print(model.summary())
    return model

@step
def baseline_model_step(X_train: pd.DataFrame, y_train: pd.Series, k: int = 10):
    """
    Trains and validates a baseline dummy model.
    """
    model = DummyRegressor(strategy='mean')
    model.fit(X_train, y_train)
    mse_scorer = make_scorer(mean_squared_error)
    mse_scores = cross_val_score(model, X_train, y_train, cv=k, scoring=mse_scorer)
    rmse_scores = sqrt(mse_scores)
    print(f"Baseline MSE: {mse_scores.mean()}")
    print(f"Baseline RMSE: {rmse_scores.mean()}")
    return model, mse_scores.mean(), rmse_scores.mean()

import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro


@step
def model_refinement_step(model, df: pd.DataFrame):
    """
    Refines linear regression model: removes insignificant variables, checks multicollinearity,
    normality, homoscedasticity, and validates with K-Fold CV.
    """
    class ModelRefinement:
        def __init__(self, model, data):
            self.model = model
            self.data = data
            self.predictors = [x for x in self.model.model.exog_names if x != 'const']
            self.target = self.model.model.endog_names
            self.rmse = None

        def remove_insignificant_vars(self, alpha=0.05):
            summary_df = pd.DataFrame(self.model.summary().tables[1].data)
            summary_df.columns = summary_df.iloc[0]
            summary_df = summary_df.drop(0).set_index(summary_df.columns[0])
            summary_df['P>|t|'] = summary_df['P>|t|'].astype(float)
            self.predictors = [var for var in self.predictors if summary_df.loc[var, 'P>|t|'] < alpha]
            return self.predictors

        def check_multicollinearity(self):
            exog = sm.add_constant(self.data[self.predictors])
            vif = pd.Series([variance_inflation_factor(exog.values, i) for i in range(exog.shape[1])],
                            index=exog.columns)
            print("VIF:")
            print(vif)

        def check_normality_of_residuals(self):
            residuals = self.model.resid
            qqplot(residuals, line='s')
            plt.show()
            stat, p = shapiro(residuals)
            print('Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (stat, p))
            alpha = 0.05
            print('Gaussian' if p > alpha else 'Not Gaussian')

        def check_homoscedasticity(self):
            residuals = self.model.resid
            plt.scatter(self.model.predict(), residuals)
            plt.axhline(y=0, color='red')
            plt.xlabel('Predicted')
            plt.ylabel('Residual')
            plt.title('Residual vs Predicted')
            plt.show()

        def validate(self, k=10):
            from sklearn.model_selection import KFold
            from sklearn.metrics import mean_squared_error
            X = sm.add_constant(self.data[self.predictors])
            y = self.data[self.target]
            kf = KFold(n_splits=k)
            errors = []
            for train_idx, test_idx in kf.split(X):
                model_cv = sm.OLS(y.iloc[train_idx], X.iloc[train_idx]).fit()
                predictions = model_cv.predict(X.iloc[test_idx])
                errors.append(mean_squared_error(y.iloc[test_idx], predictions))
            self.rmse = np.sqrt(np.mean(errors))
            return self.rmse


    refinement = ModelRefinement(model, df)
    predictors = refinement.remove_insignificant_vars(alpha=0.05)
    refinement.check_multicollinearity()
    refinement.check_normality_of_residuals()
    refinement.check_homoscedasticity()
    rmse = refinement.validate(k=10)
    print(f"Refined model RMSE: {rmse}")
    return refinement