# Capstone_5_SingaporeFlatResalePredict :european_castle:

## :page_with_curl: Project Overview
This project involves understanding and preprocessing data, conducting Exploratory Data Analysis (EDA), and developing machine learning models to address challenges in predicting resale prices. Specifically, the project focuses on building a regression model to predict the resale price of properties. Additionally, an interactive Streamlit web application will be created to facilitate real-time predictions using these models. The final step will involve deploying the web application on the Render platform for user access.

## Domain: :hotel: Real Estate 

## 🛠 Technologies Used
* Python 
* Numpy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Pickle
* Streamlit
* Render

## 📘 Approach

### Data Understanding: :page_facing_up:
Load and understand the resale flat price data through CSV files.
Check the data for consistency and completeness.

### Data Preprocessing:✂️
Loaded the resale flat price CSV into a DataFrame.
Cleaned and filled missing values, addressed outliers,and adjusted data types.
Analyzed data distribution and treated skewness.

### Exploratory Data Analysis(EDA): :bar_chart:
Understanding and visualizing the data using EDA techniques such as boxplots, histograms, and violin plots.

### Feature Engineering: :chart_with_upwards_trend:
Drop highly correlated columns using a heatmap from Seaborn.

### Model Evaluation through Regression Model: :triangular_ruler:
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R-squared (R²).

Split the dataset into training and testing sets.
Train and evaluate regression models for Predicted Resale_Price :money_with_wings:.
Pickled the trained models for deployment.

### Streamlit: :computer:
The user interface and visualization are created using the Streamlit framework.

### Render Deployment: :desktop_computer:
Deploy the web application on a hosting platform like Render for user access.


### 💻Import Packages
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle
import streamlit as st
