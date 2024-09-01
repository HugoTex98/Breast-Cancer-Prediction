import pandas as pd


def Clear(breast_cancer_data: pd.DataFrame):
    """
    Clears the data present in the provided DataFrame, leaving it empty, and displays 
    memory usage and the number of records deleted.
    
    Parameters:
        breast_cancer_data (pd.DataFrame): DataFrame containing breast cancer patient data.
    
    Returns:
        None
    """
    
    dataset = breast_cancer_data
    if not dataset.empty:
        # Para limpar os dados do DataFrame
        dataset_clear = dataset.iloc[0:0]
        print(dataset_clear)
        print('\nMemory usage: \n')
        print(dataset_clear.memory_usage()) # Mostra a memória usada por cada variável no DataFrame
        print('\nRecords deleted: ', len(dataset) - len(dataset_clear))
        
    else:
        print('\nNot possible to clear the DataFrame!')
     