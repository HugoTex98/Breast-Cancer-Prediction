import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from __init__ import Sort, Path, logging


def Correlation(breast_cancer_dataset: pd.DataFrame, results_folder: Path) -> pd.DataFrame:
    """
    Analyzes and visualizes the correlation between features in the breast cancer dataset 
    and the classification label, and returns a cleaned dataset.

    Parameters:
        breast_cancer_dataset (pd.DataFrame): DataFrame containing breast cancer patient data.
        results_folder (Path): The path to the newly created folder, or the existing folder 
                               if it was already present, to store the run results.

    Returns:
        pd.DataFrame: A cleaned version of the input DataFrame, with irrelevant columns removed, 
        sorted by `patient_id`, and with the `classification` column converted to numerical values.
        Additionally, a heatmap is displayed showing the correlation of each feature with the 
        `classification` label.
    """
    
    # Use `Sort` function to clean the dataset
    cleaned_breast_cancer_dataset = Sort(breast_cancer_dataset)
        
    # Drop [study_id, series, lesion_id, segmentation_id, image_view, mammography_type] columns from 
    # cleaned_breast_cancer_dataset DataFrame except 'patient_id' (important for SPLITSCALE function)
    cleaned_breast_cancer_dataset = cleaned_breast_cancer_dataset.drop(['study_id', 'series', 'lesion_id',
                                                                        'segmentation_id', 'image_view', 
                                                                        'mammography_type'], axis = 1)
    
    # Copy of cleaned_breast_cancer_dataset without "patient_id" because its not necessary for next steps
    corr_breast_cancer_dataset = cleaned_breast_cancer_dataset.copy()
    corr_breast_cancer_dataset.drop(['patient_id'], axis = 1, inplace = True)
    
    # Correlation of all features related to the label "classification" 
    # Lets use Spearman correlation in the case of non-linear correlations between features and label   
    correlation_breast_cancer = corr_breast_cancer_dataset.loc[:,
                                                               corr_breast_cancer_dataset.columns != 'classification'].corrwith(corr_breast_cancer_dataset['classification'],
                                                                                                                                method = 'spearman')   
    correlation_breast_cancer = pd.DataFrame(correlation_breast_cancer, columns=['classification'])
                                                                                                            
    # Correlation heatmap of all features related to the label "classification"
    plt.figure()
    sns.heatmap(correlation_breast_cancer, cmap = "crest", annot = True)
    plt.savefig(Path.joinpath(results_folder, "Correlation_Heatmap.png"))
    logging.info("Saved Correlation_Heatmap.png!")
    plt.show(block=False)  # Non-blocking call
    
    return cleaned_breast_cancer_dataset