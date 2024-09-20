from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from __init__ import Splitscale, Path, logging


def AnnModel(breast_cancer_dataset: pd.DataFrame,
             results_folder: Path) -> Tuple[np.ndarray, np.ndarray, float, MLPClassifier]:
    """ 
    Trains an Artificial Neural Network (ANN) model on the breast cancer dataset and evaluates its performance.

    Parameters:
        breast_cancer_dataset (pd.DataFrame): DataFrame containing breast cancer patient data.
        results_folder (Path): The path to the newly created folder, or the existing folder 
                               if it was already present, to store the run results.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, MLPClassifier]:
            - y_test (np.ndarray): Array of true labels ('classification') for the test dataset.
            - y_pred (np.ndarray): Array of predicted labels ('classification') from the ANN model.
            - accuracy (float): Accuracy score of the ANN model on the test dataset.
            - modelo_MLP (MLPClassifier): The trained Multi-layer Perceptron (MLP) model.
    """
    
    # Use of SPLITSCALE function to clean the dataset (using also SORT and CORRELATION function) and get
    # train (X_train, y_train)and test (X_test, y_test) datasets 
    X_train, y_train, X_test, y_test = Splitscale(breast_cancer_dataset, results_folder)
    
    # Create Random Search for parameters optimization
    # param_grid = {
    #     'hidden_layer_sizes': [(randint.rvs(10, 1000),),(randint.rvs(10, 1000),randint.rvs(10, 1000)), 
    #                            (randint.rvs(10, 1000),randint.rvs(10, 1000), randint.rvs(10, 1000))], # Testamos de 1 até 3 hidden layers
    #     'activation': ['relu', 'tanh', 'identity'],
    #     'solver': ['adam', 'sgd', 'lbfgs'],
    #     'learning_rate_init': uniform(0.001, 0.1),
    # }   
    # modelo_MLP = MLPClassifier(max_iter = 600, random_state = 42)
    # search = RandomizedSearchCV(modelo_MLP, param_grid, scoring = 'roc_auc', cv = 10) # Pode-se otimizar a search para outra métrica!!
    # search.fit(X_train.values, y_train.squeeze())
    # melhores_params = search.best_params_  # Para ir buscar os melhores parametros do modelo gerados pela RandomSearch: search.best_params_['variavel a querer']
        
    # Create MLP model through Random Search parameters (already ran)
    modelo_MLP = MLPClassifier(hidden_layer_sizes = (704, 564), activation = 'tanh',
                                solver = 'sgd', 
                                learning_rate_init = 0.05035503688561887, max_iter = 600, random_state = 42) 
    
    # Fit MLP model in train dataset 
    modelo_MLP.fit(X_train.values, y_train)
    
    # Prediction on test dataset
    y_pred = modelo_MLP.predict(X_test.values)
    
    # First and last 5 predictions of MLP model
    logging.info('\nFirst 5 ground-truths: \n', y_test[:5])
    logging.info('\nFirst 5 predictions of MLP model: \n', y_pred[:5])
    logging.info('\nLast 5 ground-truths: \n', y_test[-5:])
    logging.info('\nLast 5 predictions of MLP model: \n', y_pred[-5:])
    
    # Evaluate model in test dataset
    accuracy = modelo_MLP.score(X_test.values, y_test)
    logging.info('\nAccuracy of MLP model: {:.2f}% \n'.format(accuracy * 100))

    # Save model parameters:
    with open(f"{results_folder}/ANN_params.txt", "w") as params_file:
        # `get_params()` return a dict, we need to convert to string
        params_file.write(str(modelo_MLP.get_params()))
        
    return y_test, y_pred, accuracy, modelo_MLP
    

def Ann_to_Metrics(breast_cancer_dataset: pd.DataFrame,
                   results_folder: Path) -> Tuple[np.ndarray, np.ndarray, float, MLPClassifier]:
    """ 
        Trains an Artificial Neural Network (ANN) model on the breast cancer dataset and evaluates its performance.

        Parameters:
            breast_cancer_dataset (pd.DataFrame): DataFrame containing breast cancer patient data.
            results_folder (Path): The path to the newly created folder, or the existing folder 
                               if it was already present, to store the run results.

        Returns:
            Tuple[np.ndarray, np.ndarray, float, MLPClassifier]:
                - y_test (np.ndarray): Array of true labels ('classification') for the test dataset.
                - y_pred (np.ndarray): Array of predicted labels ('classification') from the ANN model.
                - accuracy (float): Accuracy score of the ANN model on the test dataset.
                - modelo_MLP (MLPClassifier): The trained Multi-layer Perceptron (MLP) model.
    """

    # Use of SPLITSCALE function to clean the dataset (using also SORT and CORRELATION function) and get
    # train (X_train, y_train)and test (X_test, y_test) datasets 
    X_train, y_train, X_test, y_test = Splitscale(breast_cancer_dataset, results_folder)
    
    # Create Random Forest model through Random Search parameters (already ran)
    modelo_MLP = MLPClassifier(hidden_layer_sizes = (704, 564), activation = 'tanh',
                                solver = 'sgd', 
                                learning_rate_init = 0.05035503688561887, max_iter = 600, random_state = 42) 
    
    # Fit MLP model in train dataset 
    modelo_MLP.fit(X_train.values, y_train)
    
    # Prediction on test dataset
    y_pred = modelo_MLP.predict(X_test.values)
    
    # Evaluate model in test dataset
    accuracy = modelo_MLP.score(X_test.values, y_test)
    logging.info('\nAccuracy of MLP model: {:.2f}% \n'.format(accuracy * 100))

    # Save model parameters:
    with open(f"{results_folder}/ANN_params.txt", "w") as params_file:
        # `get_params()` return a dict, we need to convert to string
        params_file.write(str(modelo_MLP.get_params()))
        
    return y_test, y_pred, accuracy, modelo_MLP
