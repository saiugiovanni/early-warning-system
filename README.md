# Early Warning System for Financial Markets

This repository contains a machine learning project developed during my MSc in Mathematical Engineering (Quantitative Finance).

The objective of the project is to build an Early Warning System capable of detecting periods of financial market stress using macro-financial indicators, statistical analysis and machine learning techniques.

The project combines data preprocessing, anomaly detection, classification models, feature importance analysis and portfolio-based evaluation.

## Main objectives

- Analyse macro-financial indicators linked to market stress
- Detect abnormal market conditions using machine learning
- Build predictive early warning signals for financial crises
- Perform feature engineering and time series preprocessing
- Compare different machine learning approaches
- Interpret model predictions using SHAP values
- Evaluate model performance through classification metrics
- Analyse portfolio allocation under risk-on / risk-off regimes

## Repository structure

- `Project.ipynb`  
  Main notebook containing the complete analysis, modelling workflow and visualisations.

- `external_functions.py`  
  Utility functions used for:
  - threshold optimisation,
  - ROC analysis,
  - confusion matrices,
  - F1-score optimisation,
  - SHAP interpretability plots.

- `data/ews_dataset.xlsx`  
  Dataset containing macro-financial indicators used for the analysis.

- `data/sp500_total_return.xlsx`  
  S&P500 benchmark dataset used for portfolio comparison and performance evaluation.

## Machine learning workflow

The project includes:

### Data preprocessing

- Missing value handling
- Time series transformation
- Feature engineering
- Stationarity analysis
- Scaling and normalisation

### Models explored

- Logistic Regression
- Random Forest
- Gaussian Copula
- Autoencoder

### Model evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC curves
- AUC analysis
- Confusion matrices

### Explainability

The project uses SHAP (SHapley Additive exPlanations) to analyse feature contributions and improve model interpretability.

### Portfolio analysis

A simple risk-on / risk-off portfolio allocation framework is implemented to compare strategy performance against the S&P500 benchmark.

## Technologies used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SHAP

## How to run the project

Install the required Python packages:

```bash
pip install -r requirements.txt
