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
* **Load and Understand Data:** Begin by loading the resale flat price data from CSV files. This step involves reading the data into a Pandas DataFrame and performing an initial review to understand its structure and content.
* **Ensure Data Consistency and Completeness:** Check for any inconsistencies or missing data points to ensure the dataset is complete and reliable for analysis.

### Data Preprocessing:✂️
* **Load CSV into DataFrame:** Import the CSV file into a Pandas DataFrame for easy manipulation and analysis.
* **Clean Data:** Remove or handle any erroneous or irrelevant data points.
* **Impute Missing Values:** Fill in any missing values to maintain the integrity of the dataset.
* **Address Outliers:** Identify and appropriately handle outliers that could skew the analysis.
* **Adjust Data Types:** Ensure that all columns have the correct data types for efficient processing.
* **Analyze Data Distribution and Correct Skewness:** Examine the distribution of the data and apply transformations if necessary to address any skewness.

### Exploratory Data Analysis(EDA): :bar_chart:
**Use Boxplots, Histograms, and Violin Plots:** Visualize the data to uncover underlying patterns, trends, and distributions. These plots help in understanding the spread and central tendency of the data, as well as identifying any anomalies.

### Feature Engineering: :chart_with_upwards_trend:
**Drop Highly Correlated Columns:** Use a heatmap from Seaborn to identify and drop columns that are highly correlated. This step reduces redundancy and multicollinearity, improving model performance.

### Model Evaluation through Regression Model: :triangular_ruler:
**Metrics**: MAE, MSE, RMSE, R²: Evaluate model performance using these key metrics:
* **Mean Absolute Error (MAE):** Measures the average magnitude of errors in the predictions.
* **Mean Squared Error (MSE):** Measures the average squared difference between observed and predicted values.
* **Root Mean Squared Error (RMSE):** The square root of MSE, providing error magnitude in the same units as the target variable.
* **R-squared (R²)**: Indicates the proportion of variance in the dependent variable explained by the model.
* **Split Dataset into Training and Testing Sets:** Divide the dataset into two parts: one for training the model and one for testing its performance for Predicted Resale_Price :money_with_wings:..
* **Train and Evaluate Regression Models:** Train different regression models and evaluate their performance based on the above metrics.
* **Serialize Trained Models using Pickle:** Save the trained models to disk using Pickle for later use in the web application.

### Streamlit: :computer:
**Develop UI and Visualization**: Use the Streamlit framework to create an interactive user interface that allows users to input data and view predictions. Include visualizations to enhance user experience.

### Render Deployment: :desktop_computer:
**Deploy Web Application on Render**:  Use Render to make our web application accessible on the internet so users can access it from anywhere.

### Live Application 🌐

Access the live application [here](https://capstone-5-singaporeflatresalepredict.onrender.com).


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
