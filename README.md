# Breast Cancer Prediction Project

## Project Overview

The goal of this project is to develop a Python-based program that processes, visualizes, and classifies data from a dataset containing indicators of breast cancer patients. The project involves exploring various Machine Learning models to predict (diagnose) whether a patient is likely to have breast cancer.

## Objectives

- **Data Extraction and Visualization**: Extract and visualize key characteristics from a dataset of breast cancer indicators.
- **Model Development**: Develop and compare multiple classification models (SVM, Random Forest, ANN/DNN) to predict whether a patient is likely to have breast cancer.
- **Performance Evaluation**: Utilize appropriate metrics (e.g., Accuracy, Precision, Recall) to evaluate the performance of the models and determine the best-performing classifier.

## Dataset

The dataset used in this project is named `bcdr_f01_features.csv`, which contains 44 variables: 16 integer fields, 27 real (float) fields, and 1 string field. The dataset provides indicators collected from breast cancer patients. Here's a brief description of each variable:

 - **`patient_id`**: Identifier for each patient.
 - **`study_id`**: Identifier for each study associated with a patient.
 - **`series`**: Series number within the study.
 - **`lesion_id`**: Identifier for each lesion within a study.
 - **`segmentation_id`**: Identifier for each segmentation of a lesion.
 - **`image_view`**: The view in which the image was taken (e.g., craniocaudal, mediolateral).
 - **`mammography_type`**: Type of mammography used (e.g., screening or diagnostic).
 - **`mammography_nodule`**: Indicates the presence of a nodule (binary).
 - **`mammography_calcification`**: Indicates the presence of calcification (binary).
 - **`mammography_microcalcification`**: Indicates the presence of microcalcifications (binary).
 - **`mammography_axillary_adenopathy`**: Indicates the presence of axillary adenopathy (binary).
 - **`mammography_architectural_distortion`**: Indicates the presence of architectural distortion (binary).
 - **`mammography_stroma_distortion`**: Indicates the presence of stromal distortion (binary).
 - **`age`**: Age of the patient.
 - **`density`**: Breast density category.
 - **`i_mean`**: Mean intensity value of the image.
 - **`i_std_dev`**: Standard deviation of the image intensity.
 - **`i_maximum`**: Maximum intensity value in the image.
 - **`i_minimum`**: Minimum intensity value in the image.
 - **`i_kurtosis`**: Kurtosis of the image intensity distribution.
 - **`i_skewness`**: Skewness of the image intensity distribution.
 - **`s_area`**: Area of the segmented region.
 - **`s_perimeter`**: Perimeter of the segmented region.
 - **`s_x_center_mass`**: X-coordinate of the center of mass of the segmented region.
 - **`s_y_center_mass`**: Y-coordinate of the center of mass of the segmented region.
 - **`s_circularity`**: Circularity of the segmented region.
 - **`s_elongation`**: Elongation of the segmented region.
 - **`s_form`**: Form factor of the segmented region.
 - **`s_solidity`**: Solidity of the segmented region.
 - **`s_extent`**: Extent (ratio of area to bounding box area) of the segmented region.
 - **`t_energ`**: Texture energy of the segmented region.
 - **`t_contr`**: Texture contrast of the segmented region.
 - **`t_corr`**: Texture correlation of the segmented region.
 - **`t_sosvh`**: Sum of squares variance of the texture.
 - **`t_homo`**: Texture homogeneity of the segmented region.
 - **`t_savgh`**: Sum average of the texture.
 - **`t_svarh`**: Sum variance of the texture.
 - **`t_senth`**: Sum entropy of the texture.
 - **`t_entro`**: Entropy of the texture.
 - **`t_dvarh`**: Difference variance of the texture.
 - **`t_denth`**: Difference entropy of the texture.
 - **`t_inf1h`**: First information measure of correlation.
 - **`t_inf2h`**: Second information measure of correlation.
 - **`classification`**: Classification of the lesion as either "Malign" (malignant) or "Benign".

## Program

The program implements a command-line interface (CLI) with the following functionalities:

1. **`LOAD`**: Load a specified dataset file and display summarized information.
2. **`LOADF`**: Load the provided `bcdr_f01_features.csv` file and display summarized information.
3. **`CLEAR`**: Clear the loaded data from memory.
4. **`QUIT`**: Exit the program.
5. **`DESCRIBE`**: Provide a statistical summary of the numerical data and count benign and malignant cases.
6. **`SORT`**: Sort the data by patient ID, handle missing values, and encode the classification labels.
7. **`CORRELATION`**: Remove irrelevant features and visualize correlations using a heatmap.
8. **`SPLITSCALE`**: Split the dataset into training and test sets and scale features.
9. **`SVM`**: Train and test a Support Vector Machine (SVM) classifier (already fine-tuned with Random Search).
10. **`RANDOMFOREST`**: Train and test a Random Forest classifier (already fine-tuned with Random Search).
11. **`ANN`**: Train and test an Artificial Neural Network (ANN) classifier (already fine-tuned with Random Search).
12. **`METRICS`**: Evaluate and display classification performance metrics (Confusion Matrix, Accuracy, Precision, Recall) for all the models developed.

## Project Structure

- **/scripts**: Contains the main script `main.py` that implements the above functionalities.
- **/data**: Directory to store datasets.
- **/results**: Directory to save results for each run, such as visualizations and model outputs.
- **Dockerfile**: Instructions to containerize the application.
- **requirements.txt**: List of Python dependencies.
- **README.md**: This file.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/HugoTex98/Breast-Cancer-Prediction.git
    cd Breast-Cancer-Prediction
    ```
2. **Create a virtual environment and activate it:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. **Install the required dependencies:**

The code is written in Python and requires the following libraries:

- Pandas
- Requests
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

    ```bash
    pip install -r requirements.txt
    ```

## Building and Running the Docker Container

To build and run the Docker container for this project, follow the steps below:

### 1. Build the Docker Image

First, navigate to the root directory of the project (where `Dockerfile` is located) and run the following command to build the Docker image:

```bash
docker build -t breast-cancer-prediction .
```

This command builds the Docker image using the instructions in the Dockerfile and tags it as breast-cancer-prediction.

### 2. Run the Docker Container

Once the image is built, you can run the Docker container using the following command:

```bash
docker run -it --rm breast-cancer-prediction
```

 - **`-it`**: Runs the container in interactive mode, allowing you to interact with the terminal inside the container.
 - **`--rm`**: Automatically removes the container once it stops running, keeping your environment clean.
 - **`breast-cancer-prediction`**: The name of the Docker image you built.

Since the project is not a web application and does not expose any ports, the output from the script will be directly visible in the terminal where the container runs. If the script generates output files, they will be saved in the container's file system.

To stop the container while it’s running, you can do so by pressing Ctrl + C in the terminal where the container is running.

## Usage

Run the program using the command-line interface:

```bash
python main.py
```
Follow the prompts to load data, process it, and visualize various aspects related to heart disease indicators.

## Future Improvements

In the future, I plan to implement the following improvements and features to enhance the functionality and performance of this project:

1. **New branch**: 
   - Creation of a new branch for new developments.
  
2. **Store models parameters**
   - Store models parameters in the results.

3. **Expanded Model Selection**: 
   - Integrate additional machine learning models, such as Gradient Boosting Machines (GBM) or XGBoost, to compare performance with the current models.

4. **Hyperparameter Optimization**:
   - Implement Optuna to optimize the performance of the classifiers.

5. **Data Augmentation**:
   - Apply data augmentation techniques (SMOTE) to increase the dataset and potentially improve model accuracy.

6. **User Interface**:
   - Develop a simple graphical user interface (GUI) to make the program more user-friendly, allowing users to load data, select models, and view results without needing to interact directly with the command line.

7. **Containerization and Deployment**:
   - Refine the Docker setup to allow for easy deployment on cloud platforms like AWS or Azure, including CI/CD pipeline integration.

8. **Testing and Validation**:
   - Implement unit tests and continuous integration (CI) pipelines to ensure code quality and detect potential issues before deployment (maybe ML Flow for model monitoring).

9. **Enhanced Visualization**:
    - Add more advanced visualization options, including 3D plots or interactive dashboards, to provide deeper insights into the data and model outputs.