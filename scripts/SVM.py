from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from __init__ import Splitscale, Path, logging


def SvmModel(breast_cancer_dataset: pd.DataFrame,
             results_folder: Path) -> Tuple[np.ndarray, np.ndarray, float, SVC]:
    """ 
    Trains a Support Vector Machine (SVM) model on the breast cancer dataset and evaluates its performance.

    Parameters:
        breast_cancer_dataset (pd.DataFrame): DataFrame containing breast cancer patient data.
        results_folder (Path): The path to the newly created folder, or the existing folder 
                               if it was already present, to store the run results.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, SVC]:
            - y_test (np.ndarray): Array of true labels ('classification') for the test dataset.
            - y_pred (np.ndarray): Array of predicted labels ('classification') from the SVM model.
            - accuracy (float): Accuracy score of the SVM model on the test dataset.
            - modelo_SVM (SVC): The trained SVM model.
    """
    
    # Use of SPLITSCALE function to clean the dataset (using also SORT and CORRELATION function) and get
    # train (X_train, y_train)and test (X_test, y_test) datasets 
    X_train, y_train, X_test, y_test = Splitscale(breast_cancer_dataset, results_folder)
    
    # Create Random Search for parameters optimization
    # param_grid = {
    #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #     'C': uniform(0, 10),
    #     'gamma': uniform(0, 10),
    # }   
    # modelo_SVM = SVC(random_state = 42)
    # search = RandomizedSearchCV(modelo_SVM, param_grid, n_iter = 20, scoring = 'roc_auc', cv = 10) # Pode-se otimizar a search para outra métrica!!
    # search.fit(X_train.values, y_train.squeeze())
    # melhores_params = search.best_params_ # Para ir buscar os melhores parametros do modelo gerados: search.best_params_['variavel a querer']
    
    # Create SVM model through Random Search parameters (already ran)   
    modelo_SVM = SVC(kernel='rbf', C = 2.645942744979508, gamma = 0.26312360995393913) 
    
    # Fit SVM in train dataset
    modelo_SVM.fit(X_train, y_train)
    
    # Prediction on test dataset
    y_pred = modelo_SVM.predict(X_test)
    
    # First and last 5 predictions of SVM model
    logging.info('\nFirst 5 ground-truths: \n', y_test[:5])
    logging.info('\nFirst 5 predictions of SVM model: \n', y_pred[:5])
    logging.info('\nLast 5 ground-truths: \n', y_test[-5:])
    logging.info('\nLast 5 predictions of SVM model: \n', y_pred[-5:])
    
    # Evaluate model in test dataset
    accuracy = modelo_SVM.score(X_test, y_test)
    logging.info('\nAccuracy of SVM model: {:.2f}% \n'.format(accuracy * 100))

    # Save model parameters:
    with open(f"{results_folder}/SVM_params.txt", "w") as params_file:
        # `get_params()` return a dict, we need to convert to string
        params_file.write(str(modelo_SVM.get_params()))
    
    return y_test, y_pred, accuracy, modelo_SVM


def Svm_to_Metrics(breast_cancer_dataset: pd.DataFrame,
                   results_folder: Path) -> Tuple[np.ndarray, np.ndarray, float, SVC]:
    """ 
    Trains a Support Vector Machine (SVM) model on the breast cancer dataset and evaluates its performance.

    Parameters:
        breast_cancer_dataset (pd.DataFrame): DataFrame containing breast cancer patient data.
        results_folder (Path): The path to the newly created folder, or the existing folder 
                               if it was already present, to store the run results.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, SVC]:
            - y_test (np.ndarray): Array of true labels ('classification') for the test dataset.
            - y_pred (np.ndarray): Array of predicted labels ('classification') from the SVM model.
            - accuracy (float): Accuracy score of the SVM model on the test dataset.
            - modelo_SVM (SVC): The trained SVM model.
    """

    # Use of SPLITSCALE function to clean the dataset (using also SORT and CORRELATION function) and get
    # train (X_train, y_train)and test (X_test, y_test) datasets 
    X_train, y_train, X_test, y_test = Splitscale(breast_cancer_dataset, results_folder)
        
    # Create SVM model through Random Search parameters (already ran)   
    modelo_SVM = SVC(kernel='rbf', C = 2.645942744979508, gamma = 0.26312360995393913) 
    
    # Fit SVM in train dataset
    modelo_SVM.fit(X_train, y_train)
    
    # Prediction on test dataset
    y_pred = modelo_SVM.predict(X_test)
    
    # Evaluate model in test dataset
    accuracy = modelo_SVM.score(X_test, y_test)
    logging.info('\nAccuracy of SVM model: {:.2f}% \n'.format(accuracy * 100))

    # Save model parameters:
    with open(f"{results_folder}/SVM_params.txt", "w") as params_file:
        # `get_params()` return a dict, we need to convert to string
        params_file.write(str(modelo_SVM.get_params()))
    
    return y_test, y_pred, accuracy, modelo_SVM