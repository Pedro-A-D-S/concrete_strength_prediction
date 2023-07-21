import dask.dataframe as dd
import logging
import os
from dask.distributed import Client
from typing import Tuple

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

def process_data(file_path: str) -> Tuple[dd.DataFrame, dd.DataFrame]:
    '''
    Process data from a Dask DataFrame stored in a Parquet file.

    Parameters:
        file_path (str): Path to the Parquet file containing the data.

    Returns:
        tuple: A tuple containing the Dask DataFrame for independent variables (X) and 
               the Dask DataFrame for the dependent variable (y).
    '''
    # Log the start of data processing
    logging.info(f'Processing data from file: {file_path}')
    
    try: 
        # Read the Parquet file into a Dask DataFrame
        df = dd.read_parquet(file_path)
        
        # Split data into dependent (y) and independent (X) variables
        X = df[['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']]
        y = df[['strength']] 
        
        return X, y
    except Exception as e:
        logging.error(f'It was not possible to process data: {file_path}. Error: {e}')

if __name__ == '__main__':
    # Provide the path to the Parquet file containing the data
    data_file_path = os.path.abspath('data/1-bronze/Concrete_Data_Cleaned.parquet')
    
    # Connect to the Dask cluster
    client = Client(n_workers = 4)
    
    # Process data using the function
    X, y = process_data(data_file_path)
    
    # File paths to be saved
    x_file_path = os.path.abspath('data/2-silver/X.parquet')
    y_file_path = os.path.abspath('data/2-silver/Y.parquet')
    
    # Convert Dask DataFrames to Pandas DataFrames
    X_pandas = X.compute()
    y_pandas = y.compute()
    
    # Save the Pandas DataFrames X and y to Parquet files
    X_pandas.to_parquet(x_file_path, index = False)
    y_pandas.to_parquet(y_file_path, index = False)
    
    # Log the completion of data processing
    logging.info('Data split completed.')

    # Shutdown the Dask client
    client.shutdown()
