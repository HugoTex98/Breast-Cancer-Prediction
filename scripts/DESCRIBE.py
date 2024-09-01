import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def Describe(breast_cancer_dataset: pd.DataFrame, results_folder: Path):
    """
    Provides a statistical summary of numerical columns in the breast cancer dataset 
    and visualizes the number of patients with benign and malignant lesions.
    
    Parameters:
        breast_cancer_dataset (pd.DataFrame): DataFrame containing breast cancer patient data.
    
    Returns:
        None: Outputs a statistical description of numerical data and a bar plot 
        visualizing the total number of patients with benign and malignant lesions.
    """

    dados_numericos = breast_cancer_dataset.select_dtypes(include=[np.number])
    
    # stats of numerical data
    print(dados_numericos.describe())
    
    # count patients with benign or malignant cancer
    pacientes = breast_cancer_dataset['classification'].value_counts()
    n_doentes_benignos = pacientes[0] 
    n_doentes_malignos = pacientes[1]
    print(f'\nThere are {n_doentes_benignos} patients with benign lesions and {n_doentes_malignos} patients with malignant lesions\n')
    
    # barplot visualization
    plt.figure()
    plt.bar(x = pacientes.index, height = pacientes.values, color = ['green', 'red'], width=0.5)
    for i, v in enumerate(pacientes.values):
        plt.text(i, v, str(v), color='black', fontweight='bold', ha='center', va='bottom')
    plt.title('Bar plot of benign and malignant patients')
    plt.savefig(Path.joinpath(results_folder, "LABEL_BARPLOT.png"))
    plt.show(block=False)
    