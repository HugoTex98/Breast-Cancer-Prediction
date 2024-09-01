import pandas as pd


def Sort(breast_cancer_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and sorts the breast cancer dataset, removing null values, sorting by patient ID, 
    and converting the classification labels to numerical values.

    Parameters:
        breast_cancer_dataset (pd.DataFrame): DataFrame containing breast cancer patient data.

    Returns:
        pd.DataFrame: A cleaned version of the input DataFrame, sorted by `patient_id`, 
        with rows containing null values removed and the `classification` column 
        converted to numerical values (`Benign` = 0, `Malign` = 1).
    """
    
    # Order breast_cancer_dataset DataFrame by "patient_id" (ascending)
    breast_cancer_dataset = breast_cancer_dataset.sort_values(by = ['patient_id'],
                                                              ascending = True).reset_index(drop=True)
    
    # Check null values and delete those rows (axis = 0 removes rows)
    cleaned_breast_cancer_dataset = breast_cancer_dataset.copy()
    if cleaned_breast_cancer_dataset.isnull().values.any():
        cleaned_breast_cancer_dataset = cleaned_breast_cancer_dataset.dropna(axis = 0).reset_index(drop=True)
    else:
        cleaned_breast_cancer_dataset = cleaned_breast_cancer_dataset
    
    # Convert "classification" column classes to binary: 
    # ‘Benign’ = 0 
    # ‘Malign’ = 1
    # cleaned_breast_cancer_dataset['classification'] = cleaned_breast_cancer_dataset['classification'].replace([' Benign ', ' Malign '], [0, 1], inplace=True)
    cleaned_breast_cancer_dataset['classification'] = cleaned_breast_cancer_dataset['classification'].apply(lambda x: 0 if x == ' Benign ' else 1)
    # Describe of "patient_id" and "classification" columns
    print('Columns "patient_id" and "classification" decription')
    print(cleaned_breast_cancer_dataset[['patient_id', 'classification']].describe())
    
    return cleaned_breast_cancer_dataset