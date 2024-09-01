# Breast Cancer Prediction Project

## Overview

This project is part of the Master’s program in Biomedical Technologies, under the course "Information Visualization" for the academic year 2022-2023. The goal of this project is to develop a Python-based program that processes, visualizes, and classifies data from a dataset containing indicators of breast cancer patients. The project involves exploring various machine learning models to predict (diagnose) whether a patient is likely to have breast cancer.

## Objectives

- **Data Extraction and Visualization**: Extract and visualize key characteristics from a dataset of breast cancer indicators.
- **Model Development**: Develop and compare multiple classification models (SVM, Random Forest, ANN/DNN) to predict whether a patient is likely to have breast cancer.
- **Performance Evaluation**: Utilize appropriate metrics (e.g., Accuracy, Precision, Recall) to evaluate the performance of the models and determine the best-performing classifier.

## Dataset

The dataset used in this project is named `bcdr_f01_features.csv`, which contains 44 variables: 16 integer fields, 27 real (float) fields, and 1 string field. The dataset provides indicators collected from breast cancer patients.

## Features

The program includes the following functionalities:

1. **LOAD**: Load a specified dataset file and display summarized information.
2. **LOADF**: Load the provided `bcdr_f01_features.csv` file and display summarized information.
3. **CLEAR**: Clear the loaded data from memory.
4. **QUIT**: Exit the program.
5. **DESCRIBE**: Provide a statistical summary of the numerical data and count benign and malignant cases.
6. **SORT**: Sort the data by patient ID, handle missing values, and encode the classification labels.
7. **CORRELATION**: Remove irrelevant features and visualize correlations using a heatmap.
8. **SPLITSCALE**: Split the dataset into training and test sets and scale features.
9. **SVM**: Train and test a Support Vector Machine (SVM) classifier.
10. **RANDOMFOREST**: Train and test a Random Forest classifier.
11. **ANN**: Train and test an Artificial Neural Network (ANN) classifier.
12. **METRICS**: Evaluate and display classification performance metrics (Confusion Matrix, Accuracy, Precision, Recall).

## Project Structure

- **/scripts**: Contains the main script `main.py` that implements the above functionalities.
- **/data**: (optional) Directory to store datasets.
- **/results**: (optional) Directory to save results such as visualizations and model outputs.
- **Dockerfile**: Instructions to containerize the application.
- **docker-compose.yml**: (optional) Configuration to run the application with Docker Compose.
- **requirements.txt**: List of Python dependencies.
- **README.md**: This file.

## Requirements

The code is written in Python and requires the following libraries:

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- TensorFlow (optional, for deep learning models)

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
