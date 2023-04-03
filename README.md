# Project: Udacity Machine Learning Engineer Nanodegree Capstone
#### Ashley Perez

## Exploratory Data Analysis (EDA) Dependencies
### Packages used for EDA included the pandas python package, the ProfileReport package from pandas_profiling
These packages were used to transition the 'csv' files to dataframes to clean the data and cast data types to the appropiate type. Profile reports were generated for before and after the data was cleaned.

## Model Development
### Packages used for model development include: autogluon, pandas, scikitlearn
This required installatin of pip, pydantic, setuptools wheel, mxnet, bokeh, and autogluon. Further, pandas, tqdm, and TabularPredictor were imported. Autogluon was the choice AutoML used to select the best model for solving the problem.

## Model Hyperparameter Optimization
### Packages for model hyperparameter optimization included: sagemaker ipywidgets, sklearn, LabelEncoder, pandas, train_test_split, RandomizedSearchCV, lightgbm, and classification_report
The data was further cleaned and encoded for model hyperparameter optimization. The sklearn package was used primarily to develop and output best hyperparameters for given data.

## Model Training
### Packages for model training included: pandas, sklearn, train_test_split, LabelEncoder, re, lightgbm, matplotlib
The data was encoded and best hyperparameters used to train a final model. One step further, feature importance and losses were plotted to better understand the outcome of the model output. There are two training files, therefore two model types. One model was trained on data that include outcome characteristics while the other model was trained on data that does not include outcome characteristics.

## Model Productionization
### Packages for model productionization included: pandas, sklearn, re, lightgbm, sagemaker, boto3, json, image_uris, model_uris, script_uris, Estimator
The model developed was referenced to inspire two productionization sample notebooks. There is a low fidelity productionized model notebook that assumes raw data is transitioned through and may be run on a cadence. The higher fidelity notebook includes model transferred to AWS Sagemaker built-in algorithm for lightgbm and deploys the model as an endpoint that may be integrated with other applications.

## Model Data
Project requirements and benchmark model were taken from the Kaggle platform here:
https://www.kaggle.com/datasets/aaronschlegel/austin-animal-center-shelter-intakes-and-outcomes

Original and updated datasets may be pulled and referenced from here:
https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm

For datasets used specifically in this project, please refer to the 'acc_intakes_outcomes.csv' file in the project repository or zip file
