import dask.dataframe as dd
import logging
import os
from dask.distributed import Client
from typing import Tuple
from sklearn.model_selection import train_test_split as sk_train_test_split

# Set up logging
logs_dir = os.path.abspath('../../logs/')
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logs_dir, 'dask_data_processing.log'),
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up logging to display logs in the CLI
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def process_data(file_path_x: str,
                 file_path_y: str) -> Tuple[dd.DataFrame, dd.DataFrame,
                                          dd.DataFrame, dd.DataFrame]:
    '''
    Process data

    Parameters:
        file_path_x (str): Path to the Parquet file containing the independent variables (X).
        file_path_y (str): Path to the Parquet file containing the dependent variable (y).

    Returns:
        Tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame, dd.DataFrame]: A tuple containing the Dask DataFrame
        for X_train, X_test, y_train, and y_test.
    '''
    # Log the start of data processing
    logging.info(f'Processing data from files: {file_path_x}, {file_path_y}')
    
    X = dd.read_parquet(file_path_x)
    y = dd.read_parquet(file_path_y)
    
    # Apply the z-score to X
    X_scaled = (X - X.mean()) / X.std()
    
    # Convert Dask DataFrames to Pandas DataFrames
    logging.info('Converting Dask DataFrames to Pandas DataFrames...')
    X_scaled = X_scaled.compute()
    y = y.compute()

    # Train and test split
    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = sk_train_test_split(X_scaled, y, random_state=1, test_size=0.3)
    
    # Log the completion of data processing
    logging.info('Data processing completed.')
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Provide the paths to the Parquet files containing the data
    file_path_x = os.path.abspath('data/2-silver/X.parquet')
    file_path_y = os.path.abspath('data/2-silver/Y.parquet')
    
    # File paths to be saved
    file_path_X_train = os.path.abspath('data/3-gold/X_train.parquet')
    file_path_X_test = os.path.abspath('data/3-gold/X_test.parquet')
    file_path_y_train = os.path.abspath('data/3-gold/y_train.parquet')
    file_path_y_test = os.path.abspath('data/3-gold/y_test.parquet')
    
    # Connect to Dask Cluster
    logging.info('Connecting to the Dask cluster...')
    client = Client(n_workers=4)
    
    X_train, X_test, y_train, y_test = process_data(file_path_x, file_path_y)
    
    # Save the Pandas DataFrames to Parquet files
    logging.info('Saving DataFrames to Parquet files...')
    X_train.to_parquet(file_path_X_train, index=False)
    X_test.to_parquet(file_path_X_test, index=False)
    y_train.to_parquet(file_path_y_train, index=False)
    y_test.to_parquet(file_path_y_test, index=False)
    
    # Log the completion of data processing
    logging.info('Data split completed.')

    # Shut down the Dask Client
    client.shutdown()
