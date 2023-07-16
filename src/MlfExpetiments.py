import logging
from dataclasses import dataclass
import mlflow
import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator


@dataclass
class ModelEvaluator:
    '''
    Utility class for training and evaluating scikit-learn models.
    '''

    def train(self, model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
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
            mlflow.log_metric('train-accuracy', train_accuracy)
            logging.info(f'Train Accuracy: {train_accuracy:.3%}')
        except Exception as e:
            raise e

        return None

    def evaluate(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
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
            r2_score = metrics.r2_score(y_test, y_pred)
            mse_score = metrics.mean_squared_error(y_test, y_pred)

            # Log metrics
            mlflow.log_metric('r2-score', r2_score)
            mlflow.log_metric('mse', mse_score)

            # Print and log metrics
            logging.info('R2 Score: {:.3f}'.format(r2_score))
            logging.info('-' * 30)
            logging.info('MSE: {:.3f}'.format(mse_score))

            logging.info('-' * 30)
            logging.info('Metrics and artifacts logged!')
        except Exception as e:
            raise e

        return None
