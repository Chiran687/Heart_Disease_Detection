# Heart Disease Prediction

This repository contains code for predicting heart disease based on a dataset of various medical attributes. The prediction model is trained using machine learning techniques and the resulting model (`model.pkl`) can be integrated into any application for real-time predictions.

## Dataset

The dataset used for training the prediction model is located in the `Dataset` folder. It contains various attributes such as age, sex, cholesterol levels, blood pressure, etc., which are used to predict the likelihood of heart disease.

## Data Analysis

The data analysis and model training process are documented in the Jupyter Notebook `heart_disease_prediction.ipynb` located in the root folder of the repository. This notebook explores the dataset, performs data preprocessing, feature engineering, and trains a machine learning model for heart disease prediction.

## Model

The trained prediction model is serialized and saved as `model.pkl`. This model file can be loaded into any Python environment for making predictions on new data. 

## Usage

The `app` folder contains sample code demonstrating how to integrate the trained model (`model.pkl`) into a Gradio application for easy deployment. Gradio provides a simple way to create interactive web-based interfaces for machine learning models, allowing users to input data and receive predictions in real-time.

To use the heart disease prediction model in your own application:

1. Load the `model.pkl` file into your Python environment.
2. Preprocess your input data according to the requirements of the model.
3. Feed the preprocessed data into the loaded model to obtain predictions.

## Dependencies

To run the code in this repository, you'll need Python 3.x and the following libraries:

- numpy
- pandas
- scikit-learn
- gradio

You can install these dependencies using pip:

pip install numpy pandas scikit-learn gradio