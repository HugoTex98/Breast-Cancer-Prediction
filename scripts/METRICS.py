from typing import Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SVM import Svm_to_Metrics
from RANDOMFOREST import RandomForest_to_Metrics
from ANN import Ann_to_Metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, roc_curve, roc_auc_score


def Metrics(breast_cancer_dataset: pd.DataFrame,
            results_folder: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Computes and displays various metrics for different machine learning models (SVM, RandomForest, ANN) 
    using a breast cancer dataset. The function plots confusion matrices, precision, recall, accuracy bars, 
    and ROC curves for each model, and returns predictions and true labels.

    Parameters:
        breast_cancer_dataset (pd.DataFrame): DataFrame containing breast cancer patient data.
        results_folder (Path): The path to the newly created folder, or the existing folder 
                               if it was already present, to store the run results.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - y_pred_SVM (np.ndarray): Array containing predictions made by the SVM model.
            - y_pred_RF (np.ndarray): Array containing predictions made by the RandomForest model.
            - y_pred_ANN (np.ndarray): Array containing predictions made by the ANN model.
            - y_test (np.ndarray): Array containing the true labels for the test set.
    """
    
    # Get train data (X_train, y_train) and test data (X_test, y_test) of each model 
    # (SVM, RandomForest, ANN) functions
    y_test, y_pred_SVM, accuracy_SVM, modelo_SVM = Svm_to_Metrics(breast_cancer_dataset, results_folder) 
    y_test , y_pred_RF, accuracy_RF, modelo_RF = RandomForest_to_Metrics(breast_cancer_dataset, results_folder)
    y_test , y_pred_ANN, accuracy_ANN, modelo_ANN = Ann_to_Metrics(breast_cancer_dataset, results_folder)

    # Join models, preds, and metrics in a list for a results display in loop 
    modelos = [modelo_SVM, modelo_RF, modelo_ANN]
    previsoes_modelos = [y_pred_SVM, y_pred_RF, y_pred_ANN]
    accuracy_modelos = [accuracy_SVM, accuracy_RF, accuracy_ANN]
    nome_modelos = ['SVM Model', 'RF Model', 'ANN Model']

    for prev_modelo, acc_modelo, n_modelo, m in zip(previsoes_modelos, accuracy_modelos, nome_modelos, modelos):    
        # Plot of Confusion Matrix
        cm = confusion_matrix(y_test, prev_modelo)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = m.classes_)
        disp.plot()
        plt.title(f'Confusion matrix for {n_modelo}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(Path.joinpath(results_folder, f"ConfusionMatrix_{n_modelo}.png"))
        plt.show(block=False)  # Non-blocking call
        
        precision = precision_score(y_test, prev_modelo)
        recall = recall_score(y_test, prev_modelo)
        accuracy = acc_modelo
        # print(f'\nAccuracy no dataset de teste do:{:.2f}% \n'.format(accuracy * 100))
        # print(f'\nPrecision no dataset de teste do {n_modelo}: {:.2f}% \n'.format(precision))
        # print(f'\nRecall no dataset de teste do {n_modelo}: {:.2f}% \n'.format(recall))
        
        # Plot of Precision, Recall and Accuracy
        plt.figure()
        plt.bar(['Accuracy', 'Precision', 'Recall'], [accuracy, precision, recall], color = ['green', 'red', 'blue'], 
                width=0.5)
        for i, v in enumerate([accuracy, precision, recall]):
            plt.text(i, v, str(round(v, 3)), color='black', fontweight='bold', ha='center', va='bottom')
        plt.title(f'Accuracy, Precision, Recall for {n_modelo}')
        plt.savefig(Path.joinpath(results_folder, f"ACC_PREC_REC_{n_modelo}.png"))
        plt.show(block=False)  # Non-blocking call
        
        # Get AUC 
        auc = roc_auc_score(y_test, prev_modelo)
        
        # Calculate TPR (true positive rate) e FPR (false positive rate)
        fpr, tpr, thresholds = roc_curve(y_test, prev_modelo)
        
        # Plot of AUC-ROC 
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC for {n_modelo} (AUC = {round(auc,4)})')
        plt.savefig(Path.joinpath(results_folder, f"AUCROC_{n_modelo}.png"))
        plt.show(block=False)  # Non-blocking call
        
    return y_pred_SVM, y_pred_RF, y_pred_ANN, y_test
    