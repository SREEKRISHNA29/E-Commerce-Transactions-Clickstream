# E-Commerce-Transactions-Clickstream
Developed a machine learning model to predict customer purchase intent from e-commerce clickstream data. Built and compared classification models (Logistic Regression, Random Forest, XGBoost) using Python, improving marketing insights through behavior-based predictions.

Project Title: E-Commerce Purchase Prediction using Machine Learning

1. Project Overview

The goal is to predict whether a user session will end in a purchase based on their clickstream activity and behavior.

The dataset contains user interactions such as clicks, product views, and transactions.

This project demonstrates skills in data preprocessing, feature engineering, and classification modeling.

2. Project Workflow
a) Data Loading and Cleaning

Load the dataset (clickstream.csv, transactions.csv, etc.).

Handle missing values and incorrect datatypes.

Merge clickstream and transaction data on user_id or session_id.

b) Feature Engineering

Key features include:

Total number of clicks per session.

Average time between clicks.

Number of unique products viewed.

Session duration.

Whether the user added products to the cart.

Device type, browser, or region.

Create a binary target variable:

purchase = 1 if transaction occurred, else 0.

c) Data Preprocessing

Encode categorical features (OneHot or Label Encoding).

Scale numerical features using StandardScaler.

Split data into train and test sets.

d) Model Building

Train and compare multiple classifiers:

Logistic Regression – baseline model.

Random Forest Classifier – handles nonlinear relationships.

XGBoost Classifier – for performance tuning.

e) Model Evaluation

Use:

Accuracy

Precision, Recall, F1-score

ROC-AUC Score

Confusion Matrix

f) Insights

Top features influencing purchase behavior.

User patterns correlated with conversions.

Business insights to improve sales funnel.

g) Requirements

pandas
scikit-learn
xgboost
joblib
numpy

