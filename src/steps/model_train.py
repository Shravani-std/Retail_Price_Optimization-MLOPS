from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception.exception import CustomException
import sys
from abc import ABC, abstractmethod
import statsmodels.api as sm 
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor # type: ignore
import matplotlib.pyplot as plt
from numpy import sqrt
from scipy.stats import shapiro
from statsmodels.formula.api import ols # type: ignore
from statsmodels.graphics.gofplots import qqplot # type: ignore
from src.logger.logger import logging


class DataSplitter:
    def __init__(self, df: pd.DataFrame, features: List[str], target: str, test_size: float = 0.2):
        self.df = df
        self.features = features
        self.target = target
        self.test_size = test_size

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            logging.info("Starting data split process...")
            X = self.df[self.features]
            y = self.df[self.target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False
            )
            logging.info(f"Data split completed: Train shape {X_train.shape}, Test shape {X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error while splitting data")
            raise CustomException(e, sys)


# class InteractionEffects:

#     def __init__(self, data: pd.DataFrame):
#         """
#         Args:
#         data: pandas DataFrame, the data which might contain interacting variables.
#         """
#         self.data = data.copy()

#     def add_interaction(self, var1: str, var2: str):
#         """Adds an interaction term to the data.

#         Args:
#         var1: str, name of the first interacting variable
#         var2: str, name of the second interacting variable

#         Returns: 
#         pandas DataFrame, the data with the added interaction term.
#         """
#         interaction_term = self.data[var1] * self.data[var2]
#         self.data[f'{var1}:{var2}'] = interaction_term

#     def get_data(self):
#         """Returns the data with interaction terms.

#         Returns: 
#         pandas DataFrame, the data with the added interaction terms.
#         """
#         return self.data



class Model(ABC):
    """Abstract class for models."""

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self, k: int):
        pass


class LinearRegressionModel(Model):
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None

    def train(self):
        try:
            logging.info("Training Linear Regression model...")
            X_train = sm.add_constant(self.X_train)
            self.model = sm.OLS(self.y_train, X_train).fit()
            logging.info("Linear Regression model training completed successfully.")
            return self.model
        except Exception as e:
            logging.error("Error occurred during Linear Regression training.")
            raise CustomException(e, sys)

    def validate(self, k=10):
        raise NotImplementedError("Validation not implemented for linear regression model yet.")


class BaselineModel(Model):
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.model = DummyRegressor(strategy="mean")

    def train(self):
        try:
            logging.info("Training Baseline (DummyRegressor) model...")
            self.model.fit(self.X_train, self.y_train)
            logging.info("Baseline model training completed successfully.")
        except Exception as e:
            logging.error("Error occurred during Baseline model training.")
            raise CustomException(e, sys)

    def validate(self, k=10):
        try:
            logging.info("Validating Baseline model...")
            mse_scorer = make_scorer(mean_squared_error)
            mse_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=k, scoring=mse_scorer)
            rmse_scores = sqrt(mse_scores)
            logging.info(f"Baseline validation completed. MSE: {mse_scores.mean()}, RMSE: {rmse_scores.mean()}")
            print(f"Baseline MSE: {mse_scores.mean()}")
            print(f"Baseline RMSE: {rmse_scores.mean()}")
        except Exception as e:
            logging.error("Error occurred during Baseline model validation.")
            raise CustomException(e, sys)


class ModelFactory:
    @staticmethod
    def get_model(model_type: str, *args, **kwargs) -> Model:
        try:
            logging.info(f"Fetching model type: {model_type}")
            if model_type == "linear_regression":
                return LinearRegressionModel(*args, **kwargs)
            elif model_type == "baseline":
                return BaselineModel(*args, **kwargs)
            else:
                logging.error(f"Unknown model type requested: {model_type}")
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            raise CustomException(e, sys)


class ModelRefinement:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelRefinement, cls).__new__(cls)
        return cls._instance

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.predictors = [x for x in self.model.model.exog_names if x != "const"]
        self.target = self.model.model.endog_names
        self.rmse = None

    def remove_insignificant_vars(self, alpha=0.05):
        try:
            logging.info("Removing insignificant variables...")
            summary = self.model.summary().tables[1]
            summary_df = pd.DataFrame(summary.data)
            summary_df.columns = summary_df.iloc[0]
            summary_df = summary_df.drop(0).set_index(summary_df.columns[0])
            summary_df["P>|t|"] = summary_df["P>|t|"].astype(float)
            significant_vars = [var for var in self.predictors if summary_df.loc[var, "P>|t|"] < alpha]
            logging.info(f"Significant variables retained: {significant_vars}")
            self.predictors = significant_vars
            return significant_vars
        except Exception as e:
            logging.error("Error in removing insignificant variables.")
            raise CustomException(e, sys)

    def check_multicollinearity(self):
        try:
            logging.info("Checking multicollinearity (VIF)...")
            exog = sm.add_constant(self.data[self.predictors])
            vif = pd.Series([variance_inflation_factor(exog.values, i) for i in range(exog.shape[1])],
                            index=exog.columns)
            logging.info("VIF calculation completed.")
            print("Variance Inflation Factors:")
            print(vif)
        except Exception as e:
            logging.error("Error in checking multicollinearity.")
            raise CustomException(e, sys)

    def check_normality_of_residuals(self):
        try:
            logging.info("Checking normality of residuals...")
            residuals = self.model.resid
            qqplot(residuals, line="s")
            plt.show()
            stat, p = shapiro(residuals)
            logging.info(f"Shapiro test: Statistics={stat}, p={p}")
            if p > 0.05:
                print("Sample looks Gaussian (fail to reject H0)")
            else:
                print("Sample does not look Gaussian (reject H0)")
        except Exception as e:
            logging.error("Error in normality check of residuals.")
            raise CustomException(e, sys)

    def check_homoscedasticity(self):
        try:
            logging.info("Checking homoscedasticity...")
            residuals = self.model.resid
            plt.scatter(self.model.predict(), residuals)
            plt.xlabel("Predicted")
            plt.ylabel("Residual")
            plt.axhline(y=0, color="red")
            plt.title("Residual vs. Predicted")
            plt.show()
        except Exception as e:
            logging.error("Error in checking homoscedasticity.")
            raise CustomException(e, sys)

    def validate(self, k=10):
        try:
            logging.info(f"Performing {k}-Fold Cross Validation...")
            kf = KFold(n_splits=k)
            y = self.data[self.target]
            X = sm.add_constant(self.data[self.predictors])
            errors = []

            for train, test in kf.split(X):
                model = sm.OLS(y.iloc[train], X.iloc[train]).fit()
                predictions = model.predict(X.iloc[test])
                mse = mean_squared_error(y.iloc[test], predictions)
                errors.append(mse)
                logging.info(f"MSE for fold: {mse}")

            rmse = np.sqrt(np.mean(errors))
            self.rmse = rmse
            logging.info(f"Cross-validation completed. RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error("Error in cross-validation.")
            raise CustomException(e, sys)


if __name__ == "__main__": 
    df = pd.read_csv("/media/shrav/New Volume/AI/MLOPS/Retail_Price_Optimization-MLOPS-/data/retail_prices_transformed.csv") 
    df.drop(["month_year"], axis=1, inplace=True)
    X = df.drop(["qty"], axis=1) 
    y = df["qty"]
    data_splitter = DataSplitter(df, X.columns, y.name) 
    X_train, X_test, y_train, y_test = data_splitter.split()
    model = LinearRegressionModel(X_train, y_train)
    results = model.train()
    print(results.summary())
    refinement1 = ModelRefinement(results, df)
    predictors = refinement1.remove_insignificant_vars(alpha=0.05)  # removes insignificant variables 
    print(predictors) 
    X_train_significant = X_train[predictors] 
    lr_model_2 = LinearRegressionModel(X_train_significant, y_train) 
    df_with_sig_vars = pd.concat([X_train_significant, y_train], axis=1) 
    df_with_sig_vars.to_csv("/media/shrav/New Volume/AI/MLOPS/Retail_Price_Optimization-MLOPS-/data/retail_prices_transformed_sig_vars.csv", index=False) 
    model = lr_model_2.train() 
    print(model.summary())

    refinement = ModelRefinement(model, df)

    # Now you can use the methods defined in the class
    # predictors = refinement.remove_insignificant_vars(alpha=0.05)  # removes insignificant variables
    refinement.check_multicollinearity()  # checks multicollinearity among predictors
    refinement.check_normality_of_residuals()  # checks if residuals are normally distributed
    refinement.check_homoscedasticity()  # checks if residuals have constant variance
    refinement.validate(k=10)  # cross-validates the model using k-fold cross-validation

    # Assume target_var is the name of your target variable
    target_var = 'qty'

    # Calculate the target variable's standard deviation, mean and median
    std_dev = np.std(refinement.data[target_var])
    mean_value = np.mean(refinement.data[target_var])
    median_value = np.median(refinement.data[target_var])

    # Print the comparison
    print(f"Standard Deviation of {target_var}: {std_dev}")
    print(f"Mean of {target_var}: {mean_value}")
    print(f"Median of {target_var}: {median_value}")
    print(f"RMSE of the model: {refinement.rmse}")

    baseline = BaselineModel(X, y)
    baseline.train()
    baseline.validate(k=10)

    # # Initialize the class
    # interaction_effects = InteractionEffects(df)

    # # Add interaction terms
    # interaction_effects.add_interaction('var1', 'var2')  # replace 'var1' and 'var2' with the names of your interacting variables

    # # Get the data with interaction terms
    # df_with_interaction = interaction_effects.get_data()
     