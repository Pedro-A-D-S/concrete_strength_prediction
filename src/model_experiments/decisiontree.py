import os
import numpy as np
import pandas as pd
import argparse

from modelevaluator import ModelEvaluator
from logging import info
from mlflow import log_metric, set_tag, log_param, start_run, active_run, end_run
from mlflow.sklearn import log_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# system
os.chdir('../')

# Get Data
X = pd.read_csv('../data/2-silver/X.csv')
X_train = pd.read_csv('../data/3-gold/X_train.csv')
X_test = pd.read_csv('../data/3-gold/X_test.csv')

y = pd.read_csv('../data/2-silver/y.csv')
y_train = pd.read_csv('../data/3-gold/y_train.csv')
y_test = pd.read_csv('../data/3-gold/y_test.csv')

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--min_samples_leaf', type = int, default = 1)
parser.add_argument('--min_samples_split', type = int, default = 2)
parser.add_argument('--max_features', type = int, default = None)
parser.add_argument('--max_depth', type = int, default = None)
args = parser.parse_args()

# Instances
dt_model = DecisionTreeRegressor(min_samples_leaf = args.min_samples_leaf,
                                 min_samples_split = args.min_samples_split,
                                 max_depth = args.max_depth,
                                 max_features = args.max_features)
evaluator = ModelEvaluator()

# Experiment
with start_run():
    run_name = 'Random Forest'
    set_tag('mlflow.runName', run_name)
    
    # Train the model
    evaluator.train(X_train, y_train)
    
    # Log hyperparameters
    log_param("max_depth", args.max_depth)
    log_param("min_samples_split", args.min_samples_split)
    log_param("min_samples_leaf", args.min_samples_leaf)
    log_param("max_features", args.max_features)
    
    # Perform cross-validation
    k = 20
    kfold = KFold(n_splits = k, random_state = 70, shuffle = True)
    K_results = cross_val_score(dt_model, X, y, cv = kfold)
    accuracy = np.mean(abs(K_results))
    
    # Log cross-validation-metrics
    log_metric('cv_accuracy', accuracy)
    info('cv accuracy loaded successfully.')
    
    # Log the model
    log_model(dt_model, 'random-forest')
    
    # Print the run UUID
    print('Model run: ', active_run().info.run_uuid)

# End run
end_run()