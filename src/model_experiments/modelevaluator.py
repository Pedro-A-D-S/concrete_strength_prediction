from logging import info
from mlflow import log_metric
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from mlflow import log_metric, set_tag, log_param, start_run, active_run, end_run
from mlflow.sklearn import log_model
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_score


class ModelEvaluator:
    '''
    Utility class for training and evaluating scikit-learn models.
    '''
    def __init__(self):
        pass

    def train(self, model: BaseEstimator, X_train: pd.DataFrame,
              y_train: pd.DataFrame) -> None:
        '''
        Fits a scikit-learn model.

        Parameters:
            model (BaseEstimator): A scikit-learn model object.
            X_train (pd.DataFrame): The input features for training.
            y_train (pd.DataFrame): The target variable for training.

        Returns:
            None
        '''
        try:
            model = model.fit(X_train, y_train)
            train_accuracy = model.score(X_train, y_train)
            log_metric('train-accuracy', train_accuracy)
            info(f'Train Accuracy: {train_accuracy:.2%}')
        except Exception as e:
            raise e

        return None

    def evaluate(self, model: BaseEstimator, X_test: pd.DataFrame,
                 y_test: pd.DataFrame) -> None:
        '''
        Evaluates a scikit-learn model.

        Parameters:
            model (BaseEstimator): A scikit-learn model object.
            X_test (pd.DataFrame): The input features for testing.
            y_test (pd.DataFrame): The target variable for testing.

        Returns:
            None
        '''
        try:
            # Model predictions
            y_pred = model.predict(X_test)

            # Model performance metrics
            r2_score = r2_score(y_test, y_pred)
            mse_score = mean_squared_error(y_test, y_pred)

            # Log metrics
            log_metric('r2-score', r2_score)
            log_metric('mse', mse_score)

            # Print and log metrics
            info('R2 Score: {:.2f}'.format(r2_score))
            info('MSE: {:.2f}'.format(mse_score))

            info('Metrics and artifacts logged!')
        except Exception as e:
            raise e

        return None
            
            
