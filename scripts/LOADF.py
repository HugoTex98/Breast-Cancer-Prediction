import pandas as pd


def LoadDF(diretoria_dataset: str) -> pd.DataFrame:
    """
    Allows the user to load an external dataset from a specified file path and prints 
    the first and last 5 records of the dataset.
    
    Parameters:
        diretoria_dataset (str): The file path of the .csv dataset to load.
    
    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame. If the file is not 
        found, returns an empty DataFrame.
    """
    
    if diretoria_dataset.exists():
        print('File founded! \n' )
        dataset = pd.read_csv(diretoria_dataset)
        print(dataset.info())
        print('\n First 5 records: \n')
        print(dataset.head(n=5))
        print('\n Last 5 records: \n')
        print(dataset.tail(n=5))
    else:
        dataset = pd.DataFrame({'Empty' : []})
        print('File not found...')
    
    return dataset

