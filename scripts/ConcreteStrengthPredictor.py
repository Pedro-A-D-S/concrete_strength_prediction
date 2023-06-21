import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor


class ConcreteStrengthPredictor:
    """
    Concrete strength predictor using various regression algorithms.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.dataset = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'Ada Boosting': AdaBoostRegressor(),
            'KNN Regressor': KNeighborsRegressor(),
            'Bagging Regressor': BaggingRegressor(),
            'Support Vector Regressor': SVR(),
            'XGBoost Regressor': XGBRegressor(),
            'Decision Tree Regressor': DecisionTreeRegressor()
        }

    def load_dataset(self):
        """
        Load the dataset from the provided data path.
        """
        self.dataset = pd.read_parquet(self.data_path)

    def split_dataset(self):
        """
        Split the dataset into features (X) and target variable (y),
        and further split them into training and testing sets.
        """
        self.X = self.dataset.drop('strength', axis=1)
        self.y = self.dataset['strength']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def fit_models(self):
        """
        Fit the regression models to the training data.
        """
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)

    def calculate_accuracy(self, model, X_test, y_test):
        """
        Calculate the accuracy of the given model on the test data.
        """
        return model.score(X_test, y_test)

    def evaluate_models(self, k=20):
        """
        Evaluate the regression models using k-fold cross-validation.
        """
        results = []
        kfold = KFold(n_splits=k, random_state=70, shuffle=True)

        for model_name, model in self.models.items():
            accuracy = self.calculate_accuracy(model, self.X_test, self.y_test)
            scores = cross_val_score(model, self.X, self.y, cv=kfold)
            mean_accuracy = np.mean(np.abs(scores))
            results.append({
                'Algorithm': model_name,
                'Model': model,
                'Accuracy': accuracy,
                'Mean Accuracy': mean_accuracy
            })

        return pd.DataFrame(results)


if __name__ == '__main__':
    data_path = 'data/1-bronze/Concrete_Data_Cleaned.parquet'

    predictor = ConcreteStrengthPredictor(data_path)
    predictor.load_dataset()
    predictor.split_dataset()
    predictor.fit_models()
    results_df = predictor.evaluate_models()

    print(results_df)
