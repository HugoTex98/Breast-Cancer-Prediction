import logging
from pathlib import Path 
from datetime import datetime
# Import scripts functions to be used in the main.py
from LOADF import LoadDF
from LOAD import Load
from CLEAR import Clear
from DESCRIBE import Describe
from SORT import Sort
from CORRELATION import Correlation
from SPLITSCALE import Splitscale
from SVM import SvmModel, Svm_to_Metrics
from RANDOMFOREST import RandomForestModel, RandomForest_to_Metrics
from ANN import AnnModel, Ann_to_Metrics
from METRICS import Metrics


def create_results_run_folder() -> Path:
    """
    Creates a folder within the current working directory to store results, named with the current timestamp.
    
    The folder is created inside a 'results' directory, and its name is based on the current date and time 
    in the format 'YYYYMMDD_HHMMSS'. If the folder already exists, a message is printed to indicate this.
    
    Returns:
        Path: The path to the newly created folder, or the existing folder if it was already present.
    
    Raises:
        FileExistsError: If the folder already exists (caught internally and logged).
    """
    path = Path.cwd() / "results" / datetime.now().strftime(format="%Y%m%d_%H%M%S")
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logging.error("Results folder is already created.")
    else:
        logging.info("Results folder was created!")

    return path


# Dataset directory
dataset_directory = 'https://raw.githubusercontent.com/HugoTex98/Breast-Cancer-Prediction/main/dataset/bcdr_f01_features.csv' 