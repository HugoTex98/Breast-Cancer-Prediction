import requests
import pandas as pd


def LoadDF(repo_dir: str) -> pd.DataFrame:
    """
    Allows the user to load an external dataset from a specified GitHub file path and prints 
    the first and last 5 records of the dataset.
    
    Parameters:
        repo_dir (str): The GitHub file path of the .csv dataset to load.
    
    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame. If the file is not 
        found, returns an empty DataFrame.
    """
    # HTTP request to check if the file exists
    response = requests.get(repo_dir)

    if response.status_code == 200:
        print('File founded! \n' )
        dataset = pd.read_csv(repo_dir)
        print(dataset.info())
        print('\n First 5 records: \n')
        print(dataset.head(n=5))
        print('\n Last 5 records: \n')
        print(dataset.tail(n=5))
    else:
        dataset = pd.DataFrame({'Empty' : []})
        print(f'File not found... Status code: {response.status_code}')
    
    return dataset