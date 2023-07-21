# TechCon Concrete Strength

## Introduction

Welcome to the **Concrete Strength Prediction Project**, developed by the Data Science Team at TechCon Inc. This project aims to predict the strength of concrete, enabling our company to enhance product quality, optimize production processes, and improve overall efficiency.

## Problem Statement
At TechCon Inc., accurately predicting the strength of our concrete is crucial for ensuring **high-quality** construction materials. It allows us to identify potential issues before mass production, avoid financial losses from poor productions, and guarantee customer satisfaction. Additionally, accurate strength prediction enables us to optimize the usage of inputs, reduce costs, and increase profitability.

## Dataset
We have utilized a comprehensive dataset available on Kaggle, which contains information on various factors affecting concrete strength. The dataset can be accessed here: https://www.kaggle.com/datasets/sinamhd9/concrete-comprehensive-strength.

It includes features such as cement proportions, age, and other relevant parameters, along with corresponding concrete strength measurements.

## Approach

Our Data Science Team has employed various regression algorithms to develop a concrete strength prediction model. These algorithms include:

* Linear Regression
* KNN
* Decision Tree
* Random Forest
* Gradient Boosting 
* AdaBoost Regression
* Bagging Regression
* Support Vector Regression

## Project Structure

**Notebooks:**

  **Exploratory Data Analysis (EDA) and Data Preparation:** This notebook provides insights into the dataset, performs data cleaning and preprocessing, and prepares the data for model training.

  **Model Construction and Evaluation:** This notebook presents the construction and evaluation of machine learning models using the scikit-learn and XGBoost libraries. It includes the implementation of various regression algorithms mentioned earlier and evaluates their performance using metrics such as mean absolute error, median absolute error, and R-squared score. Additionally, two data files are available in this repository:

  - The original dataset downloaded from Kaggle in CSV format.
  - A cleaned and preprocessed dataset in Parquet format, ready for use with machine learning models. 

**mlruns:** It contains the tracking information of our MLflow experiments. Each run in the `mlruns` directory corresponds to a specific experiment, capturing the hyperparameters, metrics, and artifacts logged during model training and evaluation. The `mlruns` directory is organized based on timestamped folders, making it easy to track and compare different runs. This information provides valuable insights into the experimentation process, allowing for reproducibility and transparent analysis of the models developed in this project.

**src folder:** The `src` folder contains the source code for the concrete strength prediction project. It includes the implementation of utility functions for training and evaluating machine learning models, as well as other relevant scripts and modules. This folder structure promotes modularization and maintainability of the codebase, making it easier to understand and extend the project functionality.

## Usage of this project

1. Clone this repository to your local machine:
```
git clone https://github.com/Pedro-A-D-S/concrete_strength_prediction.git
```
2. Install the required dependencies:
```
pip install src/requirements.txt
```
3. Execute the notebooks in the specified order, ensuring that the dataset and necessary files are correctly referenced.


## Dask as the Runtime Engine

In the data processing step, we have incorporated **Dask**, a parallel computing library, to handle large-scale data efficiently. With Dask, we can process data in a distributed manner, allowing us to scale our computation to multiple cores and machines seamlessly. The use of Dask enables us to leverage the power of parallel processing, making our data processing pipelines faster and more scalable.

Feel free to experiment with different regression algorithms and hyperparameter tuning to further enhance the model performance. Share your feedback and contribute to this project to help us improve and expand its capabilities.

## MLflow Integration
To ensure reproducibility and transparency in our experiments, we have integrated MLflow into our workflow. MLflow is an open-source platform that allows us to track and manage our machine learning experiments effectively. With MLflow, we can log hyperparameters, metrics, and artifacts for each experiment, making it easy to compare models, track progress, and share insights with stakeholders.

Those are the results of the MLFlow Runs:

![MLFlow Runs](/images/MLFlow-image-1.jpeg)

## Model Registry
We have registered the best performing XGBoost model in MLflow's model registry. This provides a central repository for the model, allowing other teams within TechCon Inc. to easily access and deploy it in various applications and production environments.

## Contact

For inquiries or further information, please contact me at:
 - Email: pedroalves0409@gmail.com
 - LinkedIn: https://www.linkedin.com/in/pedro-a-d-s/

## License
This project is licensed under the MIT License.
