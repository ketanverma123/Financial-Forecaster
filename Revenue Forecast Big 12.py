#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install xgboost')


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load the dataset using raw string literal
data = pd.read_csv(r'C:\Users\kavin\OneDrive\financial_data.csv')

# Assuming 'Company' is a column identifying each of the 12 companies
unique_companies = data['Company '].unique()

# Iterate over each company
for company in unique_companies:
    # Subset data for the current company
    company_data = data[data['Company '] == company].copy()

    # Create lag features for revenue
    company_data['Revenue_Lag1'] = company_data['Revenue'].shift(1)
    company_data['Revenue_Lag2'] = company_data['Revenue'].shift(2)

    # Drop rows with NaN values created by lagging
    company_data.dropna(inplace=True)

    # Define features and target
    features = ['Revenue_Lag1', 'Revenue_Lag2', 'Market Cap(in B USD)', 'Gross Profit', 'Net Income', 'EBITDA', 'Inflation Rate(in US)']
    target = 'Revenue'

    X = company_data[features]
    y = company_data[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # XGBoost model initialization
    model = XGBRegressor()

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get best model from grid search
    best_model = grid_search.best_estimator_

    # Train best model on full training data
    best_model.fit(X_train, y_train)

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error for {company}: {mse}')

    # Plot actual vs predicted revenue
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.title(f'Actual vs Predicted Revenue for {company}')
    plt.show()


# In[ ]:




