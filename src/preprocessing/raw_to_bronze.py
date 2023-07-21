import dask.dataframe as dd
import pandas as pd
import logging
import os

from dask.distributed import Client

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


def process_data(file_path: str) -> dd.DataFrame:
    '''
    Process raw data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        dd.DataFrame: Processed DataFrame.
    '''
    logging.info(f"Processing data from file: {file_path}")
    df = dd.read_csv(file_path)

    df = df.rename(columns={'Cement (component 1)(kg in a m^3 mixture)': 'cement',
                            'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'slag',
                            'Fly Ash (component 3)(kg in a m^3 mixture)': 'ash',
                            'Water  (component 4)(kg in a m^3 mixture)': 'water',
                            'Superplasticizer (component 5)(kg in a m^3 mixture)': 'superplastic',
                            'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'coarseagg',
                            'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'fineagg',
                            'Age (day)': 'age',
                            'Concrete compressive strength(MPa, megapascals) ': 'strength'})

    # Perform replacement of ',' with '.' and convert numeric columns to float64
    for column in df.columns:
        if column != 'strength':  # We skip the 'strength' column in this conversion
            df[column] = df[column].replace({',': '.'}, regex = True).astype('float64')
    
    for cols in df.columns[:-1]:
        # calculating quartiles
        Q1 = df[cols].quantile(0.25)
        Q3 = df[cols].quantile(0.75)
        # iqr range
        iqr = Q3 - Q1

        # calculating the low and high limits
        low = Q1 - 1.5 * iqr
        high = Q1 + 1.5 * iqr

        # replacing outliers with the median value
        df[cols] = df[cols].where((df[cols] >= low) & (df[cols] <= high), df[cols].median())

    return df

if __name__ == '__main__':
    data_file_path = os.path.abspath('data/0-raw-data/Concrete_Data.csv')
    client = Client(n_workers=4)

    processed_data = process_data(data_file_path)
    data_processed = processed_data.compute()

    # Create an absolute path for the Parquet file
    parquet_file_path = os.path.abspath('data/1-bronze/Concrete_Data_Cleaned.parquet')

    # Save the processed data as Parquet
    data_processed.to_parquet(parquet_file_path, index=False)

    logging.info("Data processing completed and saved to Parquet.")
