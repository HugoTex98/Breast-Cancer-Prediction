from typing import Tuple
import random
import numpy as np
import pandas as pd
from __init__ import Correlation, Path, logging


def no_overlap_split(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Splits a dataset into training and testing sets without overlapping patient IDs.

    This function performs the following steps:
        1. Extracts unique patient IDs from the input DataFrame `X`.
        2. Shuffles the patient IDs to ensure randomness.
        3. Divides the shuffled patient IDs into training and testing sets (80% train, 20% test).
        4. Filters the input DataFrames `X` and `y` based on the patient IDs for the training and testing sets.
        5. Returns the training and testing subsets of `X` and `y`.

    Parameters:
        - X (pd.DataFrame): The feature DataFrame containing a column 'patient_id'.
        - y (pd.DataFrame): The target DataFrame containing a column 'patient_id'.

    Returns:
        - Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        - X_train (pd.DataFrame): The training feature DataFrame.
        - y_train (np.ndarray): The training target array.
        - X_test (pd.DataFrame): The testing feature DataFrame.
        - y_test (np.ndarray): The testing target array.

    Notes:
        - The shuffling is done with a fixed random seed (4) for reproducibility.
        - If there are overlapping patient IDs between the train and test sets, the function will print the number of such overlaps.
    """
    ids_pacientes = list(set(X['patient_id'])) # get patient_id's
    random.seed(4) # patient_id's shuffled consistently for results
    random.shuffle(ids_pacientes) # shuffle patient_id's
    
    num_pacientes = len(ids_pacientes) 
    dividir_indexes = int(0.8 * num_pacientes)  # 80% for train and 20% to test
    ids_treino = ids_pacientes[:dividir_indexes]
    ids_teste = ids_pacientes[dividir_indexes:]  
    
    overlap_ids = list(np.intersect1d(ids_treino,ids_teste))
    logging.info(f'\nThere are {len(overlap_ids)} Patient IDs overlapped in train and test datasets \n')
    
    X_train = X[X['patient_id'].isin(ids_treino)] 
    y_train = y[y['patient_id'].isin(ids_treino)] 
    X_test = X[X['patient_id'].isin(ids_teste)] 
    y_test = y[y['patient_id'].isin(ids_teste)] 

    return X_train, y_train, X_test, y_test


def Splitscale(breast_cancer_dataset: pd.DataFrame, 
               results_folder: Path) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Splits and scales the breast cancer dataset into training and testing sets, ensuring no overlap 
    between patient IDs in the training and testing sets.

    Parameters:
        breast_cancer_dataset (pd.DataFrame): DataFrame containing breast cancer patient data.
        results_folder (Path): The path to the newly created folder, or the existing folder 
                               if it was already present, to store the run results.

    Returns:
        X_train (pd.DataFrame): DataFrame containing the features for training, excluding the `classification` column 
                                and the columns removed by the `Correlation` function.
        y_train (np.ndarray): Array containing the `classification` labels for the training set.
        X_test (pd.DataFrame): DataFrame containing the features for testing, excluding the `classification` column 
                               and the columns removed by the `Correlation` function.
        y_test (np.ndarray): Array containing the `classification` labels for the testing set.
    """

    # Use of `Correlation` function to clean the dataset
    cleaned_breast_cancer_dataset = Correlation(breast_cancer_dataset, results_folder)
    
    # Split X (features) and y (label -> 'classification')
    X = cleaned_breast_cancer_dataset.loc[:, cleaned_breast_cancer_dataset.columns != 'classification']
    y = cleaned_breast_cancer_dataset[['patient_id', 'classification']]
    
    # Normalize X features (except "patient_id")
    X = X.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.name != 'patient_id' else x)

    # Create the datasets [x_train, y_train, x_test, y_test], with 80 % to train and 20% to test.
    # Ensure that there is not overlap of patient_id's in train and test datasets (a patient is either in train or
    # test dataset) 
    X_train, y_train, X_test, y_test = no_overlap_split(X, y)
    
    # Drop "patient_id" from X_train, y_train, X_test, y_test 
    X_train = X_train.drop(['patient_id'], axis = 1)
    y_train = y_train.drop(['patient_id'], axis = 1)
    X_test = X_test.drop(['patient_id'], axis = 1)
    y_test = y_test.drop(['patient_id'], axis = 1)
    
    # Indexes not ordered, need reset
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_train = y_train.squeeze().to_numpy()
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_test = y_test.squeeze().to_numpy()
    
    return X_train, y_train, X_test, y_test