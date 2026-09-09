# End-to-End Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using a structured and production-oriented workflow, from data ingestion and validation to model training, explainability, API serving, and containerization.

The project is designed to demonstrate how a machine learning model can be developed, evaluated, packaged, and exposed as an inference service rather than remaining limited to an exploratory notebook.

## Project Overview

Customer churn prediction is a binary classification problem in which the objective is to identify customers who are likely to leave a financial service.

This project implements a complete machine learning workflow built around:

* Data ingestion and validation
* Feature engineering
* Leakage-aware preprocessing
* Feature selection
* Class imbalance handling
* Hyperparameter optimization with Optuna
* XGBoost model training
* Model evaluation
* SHAP-based explainability
* Artifact management
* Reusable inference pipeline
* REST API with FastAPI
* Docker containerization

The final system accepts raw customer information and processes it through the same feature engineering, preprocessing, and feature selection steps used during training before generating a churn probability and prediction.

## Architecture

The overall machine learning workflow is:

```text
Raw Dataset
    |
    v
Data Ingestion
    |
    v
Data Validation
    |
    v
Train / Test Split
    |
    v
Feature Engineering
    |
    v
Preprocessing
    |
    +--> Numerical: Median Imputation
    |
    +--> Categorical: Most-Frequent Imputation
    |                  + One-Hot Encoding
    |
    v
Feature Selection
    |
    v
Hyperparameter Optimization
    |
    +--> Stratified K-Fold Cross-Validation
    +--> Fold-specific Preprocessing
    +--> Fold-specific Feature Selection
    +--> SMOTETomek on Training Folds Only
    +--> XGBoost
    |
    v
Best Hyperparameters
    |
    v
Final XGBoost Training
    |
    +--> SMOTETomek
    |
    v
Model Evaluation
    |
    v
Model + Preprocessor + Selector
    |
    v
Inference Pipeline
    |
    v
FastAPI
    |
    +--> /health
    +--> /predict
    +--> /explain
    |
    v
Docker
```

## Dataset

The project uses the `Churn_Modelling.csv` dataset.

The prediction target is:

```text
Exited
```

where:

* `0` represents a customer who stayed
* `1` represents a customer who churned

Identifier columns such as:

```text
RowNumber
CustomerId
Surname
```

are explicitly excluded from the machine learning feature space.

The dataset is stored under:

```text
data/raw/Churn_Modelling.csv
```

## Feature Engineering

Feature engineering is implemented in:

```text
src/preprocessing/feature_engineering.py
```

The pipeline removes identifier columns and creates additional customer-level features, including:

* `BalancePerAge`
* `ProductsPerTenure`
* `IsActiveAndHasCard`
* `AgeGroup`
* `HasBalance`
* `MultipleProducts`

For example:

```text
BalancePerAge = Balance / Age
ProductsPerTenure = NumOfProducts / Tenure
```

Zero denominators are explicitly protected against during feature creation.

The same feature engineering logic is reused during inference to ensure consistency between training and prediction.

## Data Preprocessing

Preprocessing is implemented using a `ColumnTransformer`.

### Numerical features

Numerical variables are processed using:

```text
Median Imputation
```

### Categorical features

Categorical variables are processed using:

```text
Most-Frequent Imputation
        |
        v
One-Hot Encoding
```

The encoder uses:

```python
handle_unknown="ignore"
```

so previously unseen categorical values do not cause inference failures.

Most importantly, the preprocessing transformer is fitted only on training data and subsequently used to transform validation and test data.

This prevents information from the evaluation data from influencing preprocessing.

## Feature Selection

Feature selection is implemented using:

```text
SelectKBest
```

with:

```text
ANOVA F-value
```

The default configuration selects the top 15 features.

The selector is fitted exclusively on training data and then applied to validation, test, and inference data.

This prevents the validation/test sets from influencing feature selection.

## Class Imbalance Handling

The project uses:

```text
SMOTETomek
```

to address class imbalance.

SMOTETomek combines:

* SMOTE oversampling
* Tomek links undersampling

A critical design decision is that resampling is applied only to training data.

During cross-validation:

```text
Training Fold
    |
    v
Preprocessing
    |
    v
Feature Selection
    |
    v
SMOTETomek
    |
    v
XGBoost
```

The validation fold remains untouched.

This avoids data leakage caused by applying resampling before the cross-validation split.

## Model

The final classifier is:

```text
XGBoost Classifier
```

implemented with `XGBClassifier`.

The model is trained using the hyperparameters obtained from Optuna.

The classification objective is:

```text
binary:logistic
```

The final decision threshold is:

```text
0.5
```

Therefore:

```text
churn_probability >= 0.5
        |
        +--> Churn = 1

churn_probability < 0.5
        |
        +--> Churn = 0
```

## Hyperparameter Optimization

Hyperparameter tuning is implemented in:

```text
src/models/tuning.py
```

The project uses:

```text
Optuna
```

with a TPE sampler.

The optimization process searches over parameters including:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`
* `colsample_bytree`
* `min_child_weight`
* `gamma`
* `reg_alpha`
* `reg_lambda`

The objective is based on mean cross-validation ROC-AUC.

The cross-validation strategy uses:

```text
StratifiedKFold
```

with the configured number of folds.

Each fold independently performs preprocessing and feature selection before applying SMOTETomek and training XGBoost.

This makes the hyperparameter optimization process leakage-aware.

## Model Evaluation

The model is evaluated on the held-out test set.

The evaluation module computes:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* Classification report

The evaluation is performed after the complete training pipeline has been fitted on the training data.

The test set remains isolated from model fitting and hyperparameter optimization.

## Explainable AI with SHAP

The project integrates SHAP to provide model explanations.

The explainability module uses:

```text
SHAP TreeExplainer
```

for the final XGBoost model.

Two types of explanations are supported.

### Global explanations

Global SHAP analysis identifies the features that have the largest overall contribution to model predictions.

The project can generate:

```text
shap_feature_importance.csv
shap_feature_importance.png
```

### Individual customer explanations

For an individual prediction, the API returns the most influential features together with:

* Feature name
* Feature value when it can be mapped to the original input
* SHAP value
* Direction of impact

For example:

```json
{
  "feature": "Age",
  "value": 45,
  "shap_value": 0.31,
  "effect": "increases_churn_probability"
}
```

Positive SHAP values indicate an increase in churn probability, while negative SHAP values indicate a decrease.

The explanation is generated from the same transformed feature space used by XGBoost.

## Production Inference Pipeline

A dedicated inference pipeline is implemented in:

```text
src/models/inference_pipeline.py
```

The purpose is to guarantee that raw production data follows the same transformation sequence used during model development.

The inference flow is:

```text
Raw Customer Data
        |
        v
Feature Engineering
        |
        v
Preprocessing
        |
        v
Feature Selection
        |
        v
XGBoost
        |
        v
Churn Probability
        |
        v
Prediction
```

SMOTETomek is deliberately not applied during inference.

The inference pipeline also verifies compatibility between the fitted preprocessor and feature selector before generating predictions.

## FastAPI

The project exposes the trained model through a REST API implemented with FastAPI.

API source:

```text
api/main.py
```

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Prediction

```http
POST /predict
```

The endpoint accepts customer information and returns:

```json
{
  "prediction": 1,
  "churn": true,
  "churn_probability": 0.78
}
```

The prediction uses the explicit threshold:

```text
0.5
```

### Explainability

```http
POST /explain
```

This endpoint performs the prediction and returns SHAP-based feature contributions.

The API loads the inference pipeline and SHAP explainer during application startup and reuses them across requests.

## API Documentation

Once the application is running, FastAPI automatically provides interactive documentation.

Swagger UI:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

## Project Structure

```text
End-to-End-Churn-Prediction/
│
├── api/
│   ├── __init__.py
│   ├── dependencies.py
│   ├── main.py
│   └── schemas.py
│
├── artifacts/
│   └── .gitkeep
│
├── data/
│   ├── raw/
│   │   └── Churn_Modelling.csv
│   └── processed/
│
├── notebooks/
│   └── churn_prediction.ipynb
│
├── scripts/
│   ├── check_artifacts.py
│   ├── check_logged_models.py
│   ├── predict_customer.py
│   ├── register_model.py
│   └── test_inference.py
│
├── src/
│   ├── config/
│   │   └── config.py
│   │
│   ├── data/
│   │   ├── data_ingestion.py
│   │   └── data_validation.py
│   │
│   ├── models/
│   │   ├── evaluate.py
│   │   ├── explain.py
│   │   ├── inference_pipeline.py
│   │   ├── predict.py
│   │   ├── train.py
│   │   └── tuning.py
│   │
│   ├── preprocessing/
│   │   ├── feature_engineering.py
│   │   ├── feature_selection.py
│   │   └── preprocessing.py
│   │
│   ├── tracking/
│   │   └── mlflow_tracking.py
│   │
│   ├── utils/
│   │   └── artifact_manager.py
│   │
│   └── visualization/
│       └── plots.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── main.py
├── Makefile
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/mahrane2002/End-to-End-Churn-Prediction.git
cd End-to-End-Churn-Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The project dependencies include:

* NumPy
* Pandas
* Scikit-learn
* Imbalanced-learn
* XGBoost
* SHAP
* FastAPI
* Uvicorn
* Joblib
* SciPy
* Matplotlib
* Seaborn

## Training the Model

The main training workflow is orchestrated through:

```text
main.py
```

Run:

```bash
python main.py
```

The training workflow performs the required stages from data loading through model training and evaluation.

The resulting model artifacts are stored in the configured artifact directory.

## Running the API Locally

Start the FastAPI application with:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

## Running with Docker

The project includes a Dockerfile based on:

```text
python:3.11-slim-bookworm
```

Build the Docker image:

```bash
docker build -t churn-api .
```

Run the container:

```bash
docker run -p 8000:8000 churn-api
```

The API can then be accessed at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

The container runs the FastAPI application using Uvicorn.

## Example API Request

Example request to:

```http
POST /predict
```

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 40,
  "Tenure": 5,
  "Balance": 120000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 90000
}
```

Example response:

```json
{
  "prediction": 0,
  "churn": false,
  "churn_probability": 0.23
}
```

The exact probability depends on the trained model and stored artifacts.

## Example Explainability Request

```http
POST /explain
```

The endpoint returns the prediction together with the most influential SHAP features.

Conceptually:

```json
{
  "prediction": 1,
  "churn": true,
  "churn_probability": 0.78,
  "top_features": [
    {
      "feature": "Age",
      "value": 52,
      "shap_value": 0.31,
      "effect": "increases_churn_probability"
    },
    {
      "feature": "IsActiveMember",
      "value": 0,
      "shap_value": 0.24,
      "effect": "increases_churn_probability"
    }
  ],
  "explanation_method": "SHAP"
}
```

## Design Principles

The project follows several important machine learning engineering principles.

### No data leakage

Preprocessing and feature selection are fitted only on the appropriate training data.

During cross-validation, each fold has its own fitted preprocessing and feature-selection objects.

### Training-only resampling

SMOTETomek is applied only to training data.

Validation and test data remain untouched.

### Reproducibility

A shared random state is used across the training workflow where applicable.

### Training/inference consistency

The same feature engineering, preprocessing, and feature selection logic is reused during inference.

### Artifact-based inference

The deployed API does not retrain the model.

It loads the required trained artifacts and reconstructs the inference pipeline.

### Explainability

SHAP is integrated directly into the prediction service to provide interpretable customer-level explanations.

### Containerization

The API and its runtime dependencies can be packaged into a Docker image, providing a reproducible execution environment.

## Technology Stack

| Category                    | Technology                        |
| --------------------------- | --------------------------------- |
| Language                    | Python                            |
| Data Processing             | Pandas, NumPy                     |
| Machine Learning            | Scikit-learn                      |
| Imbalanced Learning         | Imbalanced-learn                  |
| Model                       | XGBoost                           |
| Hyperparameter Optimization | Optuna                            |
| Explainability              | SHAP                              |
| API                         | FastAPI                           |
| ASGI Server                 | Uvicorn                           |
| Serialization               | Joblib                            |
| Visualization               | Matplotlib, Seaborn               |
| Containerization            | Docker                            |
| Experiment / Model Tracking | MLflow-related project components |

## Key Machine Learning Workflow

The final training workflow can be summarized as:

```text
Dataset
  |
  v
Train/Test Split
  |
  v
Feature Engineering
  |
  v
Preprocessing
  |
  v
Feature Selection
  |
  v
Optuna + Stratified K-Fold
  |
  +--> Preprocess training fold
  +--> Preprocess validation fold
  +--> Select features from training fold
  +--> Transform validation fold
  +--> SMOTETomek on training fold
  +--> Train XGBoost
  +--> Evaluate ROC-AUC
  |
  v
Best Hyperparameters
  |
  v
Final Training
  |
  +--> SMOTETomek
  +--> XGBoost
  |
  v
Test Evaluation
  |
  v
Artifacts
```

## Inference Workflow

The production-oriented inference path is deliberately different from the training path:

```text
Raw Customer
     |
     v
Feature Engineering
     |
     v
Fitted Preprocessor
     |
     v
Fitted Feature Selector
     |
     v
Trained XGBoost
     |
     +--> Probability
     |
     +--> Prediction
     |
     v
SHAP Explanation
```

No resampling is performed during inference.

## Project Scope

This project focuses on demonstrating an end-to-end machine learning workflow and the transition from model development to model serving.

It includes the core components required to expose a trained machine learning model as a reusable API and package the service into a Docker container.

It is intended as a portfolio and engineering project demonstrating practical skills in:

* Machine learning
* Feature engineering
* Model optimization
* Explainable AI
* API development
* Model inference
* Artifact management
* Containerization
* Machine learning engineering

## Future Improvements

Possible extensions include:

* Automated unit and integration testing
* CI/CD with GitHub Actions
* Model monitoring
* Data drift detection
* Model performance monitoring
* Centralized MLflow tracking
* Remote artifact storage
* API authentication
* Production deployment on a cloud platform
* Automated model retraining
* Model registry integration
* Advanced observability and logging

## Author

**Mahrane Riahi---**

Data Science and Machine Learning Engineering

GitHub:
https://github.com/mahrane2002

## License

This project is intended for educational, portfolio, and demonstration purposes.
