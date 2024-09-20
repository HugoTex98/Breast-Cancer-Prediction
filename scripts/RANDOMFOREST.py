from typing import Tuple
import pandas as pd
import numpy as np
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from __init__ import Splitscale, Path, logging


def RandomForestModel(breast_cancer_dataset: pd.DataFrame,
                      results_folder: Path) -> Tuple[np.ndarray, np.ndarray, float, RandomForestClassifier]:
    """ 
    Trains a Random Forest model on the breast cancer dataset and evaluates its performance.

    Parameters:
        breast_cancer_dataset (pd.DataFrame): DataFrame containing breast cancer patient data.
        results_folder (Path): The path to the newly created folder, or the existing folder 
                               if it was already present, to store the run results.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, RandomForestClassifier]:
            - y_test (np.ndarray): Array of true labels ('classification') for the test dataset.
            - y_pred (np.ndarray): Array of predicted labels ('classification') from the Random Forest model.
            - accuracy (float): Accuracy score of the Random Forest model on the test dataset.
            - modelo_RF (RandomForestClassifier): The trained Random Forest model.
    """
    
    # Use of SPLITSCALE function to clean the dataset (using also SORT and CORRELATION function) and get
    # train (X_train, y_train)and test (X_test, y_test) datasets 
    X_train, y_train, X_test, y_test = Splitscale(breast_cancer_dataset, results_folder)
    
    # Create Random Search for parameters optimization
    # param_grid = {
    #     'n_estimators': np.arange(10, 1010),
    #     'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    #     'min_samples_split': np.arange(2, 20),
    # }   
    # modelo_RF = RandomForestClassifier(random_state = 42)
    # search = RandomizedSearchCV(modelo_RF, param_grid, n_iter = 20, scoring = 'roc_auc', cv = 10) # Pode-se otimizar a search para outra métrica!!
    # search.fit(X_train.values, y_train.squeeze())
    # melhores_params = search.best_params_ # Para ir buscar os melhores parametros do modelo gerados: search.best_params_['variavel a querer']
    
    # Create Random Forest model through Random Search parameters (already ran)
    modelo_RF = RandomForestClassifier(n_estimators = 250, max_depth = 30, min_samples_split = 6, random_state = 42)
    
    # Fit Random Forest model in train dataset 
    modelo_RF.fit(X_train, y_train)
    
    # Prediction on test dataset
    y_pred = modelo_RF.predict(X_test)
    
    # First and last 5 predictions of Random Forest model
    logging.info('\nFirst 5 ground-truths: \n', y_test[:5])
    logging.info('\nFirst 5 predictions of Random Forest model: \n', y_pred[:5])
    logging.info('\nLast 5 ground-truths: \n', y_test[-5:])
    logging.info('\nLast 5 predictions of Random Forest model: \n', y_pred[-5:])
    
    # Evaluate model in test dataset
    accuracy = modelo_RF.score(X_test, y_test)
    logging.info('\nAccuracy of Random Forest model: {:.2f}% \n'.format(accuracy * 100))

    # Save model parameters:
    with open(f"{results_folder}/Random_Forest_params.txt", "w") as params_file:
        # `get_params()` return a dict, we need to convert to string
        params_file.write(str(modelo_RF.get_params()))
    
    return y_test, y_pred, accuracy, modelo_RF
    

def RandomForest_to_Metrics(breast_cancer_dataset: pd.DataFrame,
                            results_folder: Path) -> Tuple[np.ndarray, np.ndarray, float, RandomForestClassifier]:
    """ 
    Trains a Random Forest model on the breast cancer dataset and evaluates its performance.

    Parameters:
        breast_cancer_dataset (pd.DataFrame): DataFrame containing breast cancer patient data.
        results_folder (Path): The path to the newly created folder, or the existing folder 
                               if it was already present, to store the run results.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, RandomForestClassifier]:
            - y_test (np.ndarray): Array of true labels ('classification') for the test dataset.
            - y_pred (np.ndarray): Array of predicted labels ('classification') from the Random Forest model.
            - accuracy (float): Accuracy score of the Random Forest model on the test dataset.
            - modelo_RF (RandomForestClassifier): The trained Random Forest model.
    """

    # Use of SPLITSCALE function to clean the dataset (using also SORT and CORRELATION function) and get
    # train (X_train, y_train)and test (X_test, y_test) datasets 
    X_train, y_train, X_test, y_test = Splitscale(breast_cancer_dataset, results_folder)
        
    # Create Random Forest model through Random Search parameters (already ran)
    modelo_RF = RandomForestClassifier(n_estimators = 250, max_depth = 30, min_samples_split = 6, random_state = 42)
    
    # Fit Random Forest model in train dataset 
    modelo_RF.fit(X_train, y_train)
    
    # Prediction on test dataset
    y_pred = modelo_RF.predict(X_test)
    
    # Evaluate model in test dataset
    accuracy = modelo_RF.score(X_test, y_test)
    logging.info('\nAccuracy of Random Forest model: {:.2f}% \n'.format(accuracy * 100))

    # Save model parameters:
    with open(f"{results_folder}/Random_Forest_params.txt", "w") as params_file:
        # `get_params()` return a dict, we need to convert to string
        params_file.write(str(modelo_RF.get_params()))
    
    return y_test, y_pred, accuracy, modelo_RF
    