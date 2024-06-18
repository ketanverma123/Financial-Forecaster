#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset using raw string literal
data = pd.read_csv(r'C:\Users\kavin\OneDrive\financial_data.csv')

# Assuming 'Company' is a column identifying each of the 12 companies
unique_companies = data['Company '].unique()

# Prepare future values for each company (example placeholders)
future_values = {
    'AAPL': {
        'Revenue_Lag1': 1000,
        'Revenue_Lag2': 900,
        'Market Cap(in B USD)': 200,
        'Gross Profit': 300,
        'Net Income': 150,
        'EBITDA': 180,
        'Inflation Rate(in US)': 2.5,
    },
    'MSFT': {
        'Revenue_Lag1': 1200,
        'Revenue_Lag2': 1100,
        'Market Cap(in B USD)': 250,
        'Gross Profit': 350,
        'Net Income': 180,
        'EBITDA': 200,
        'Inflation Rate(in US)': 2.7,
    },
    'GOOG': {
        'Revenue_Lag1': 1200,
        'Revenue_Lag2': 1100,
        'Market Cap(in B USD)': 250,
        'Gross Profit': 350,
        'Net Income': 180,
        'EBITDA': 200,
        'Inflation Rate(in US)': 2.7,
    },
    'PYPL': {
        'Revenue_Lag1': 1200,
        'Revenue_Lag2': 1100,
        'Market Cap(in B USD)': 250,
        'Gross Profit': 350,
        'Net Income': 180,
        'EBITDA': 200,
        'Inflation Rate(in US)': 2.7,
    },
    'AIG': {
        'Revenue_Lag1': 1200,
        'Revenue_Lag2': 1100,
        'Market Cap(in B USD)': 250,
        'Gross Profit': 350,
        'Net Income': 180,
        'EBITDA': 200,
        'Inflation Rate(in US)': 2.7,
    },
    'PCG': {
        'Revenue_Lag1': 1200,
        'Revenue_Lag2': 1100,
        'Market Cap(in B USD)': 250,
        'Gross Profit': 350,
        'Net Income': 180,
        'EBITDA': 200,
        'Inflation Rate(in US)': 2.7,
    },
    'SHLDQ': {
        'Revenue_Lag1': 1200,
        'Revenue_Lag2': 1100,
        'Market Cap(in B USD)': 250,
        'Gross Profit': 350,
        'Net Income': 180,
        'EBITDA': 200,
        'Inflation Rate(in US)': 2.7,
    },
    'MCD': {
        'Revenue_Lag1': 1200,
        'Revenue_Lag2': 1100,
        'Market Cap(in B USD)': 250,
        'Gross Profit': 350,
        'Net Income': 180,
        'EBITDA': 200,
        'Inflation Rate(in US)': 2.7,
    },
    'BCS': {
        'Revenue_Lag1': 1200,
        'Revenue_Lag2': 1100,
        'Market Cap(in B USD)': 250,
        'Gross Profit': 350,
        'Net Income': 180,
        'EBITDA': 200,
        'Inflation Rate(in US)': 2.7,
    },
    'NVDA': {
        'Revenue_Lag1': 1200,
        'Revenue_Lag2': 1100,
        'Market Cap(in B USD)': 250,
        'Gross Profit': 350,
        'Net Income': 180,
        'EBITDA': 200,
        'Inflation Rate(in US)': 2.7,
    },
    'INTC': {
        'Revenue_Lag1': 1200,
        'Revenue_Lag2': 1100,
        'Market Cap(in B USD)': 250,
        'Gross Profit': 350,
        'Net Income': 180,
        'EBITDA': 200,
        'Inflation Rate(in US)': 2.7,
    },
    'AMZN': {
        'Revenue_Lag1': 513.983,
        'Revenue_Lag2': 469.822,
        'Market Cap(in B USD)': 1570,
        'Gross Profit': 270,
        'Net Income': 31,
        'EBITDA': 86,
        'Inflation Rate(in US)': 3.4,
    }
  
}

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
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions for the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error for {company}: {mse}')

    # Plot actual vs predicted revenue with year annotation
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)

    # Annotate each point with the corresponding year
    for i, year in enumerate(company_data.loc[y_test.index, 'Year']):
        plt.annotate(year, (y_test.iloc[i], y_pred[i]), textcoords="offset points", xytext=(0,5), ha='center')

    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Revenue (B USD)')
    plt.ylabel('Predicted Revenue (B USD)')
    plt.title(f'Actual vs Predicted Revenue for {company}')
    plt.grid(True)
    plt.show()

    # Example: Predict future revenue for the next quarter
    # Prepare future data (assuming future values are available or placeholders)
    future_features = {
        'Revenue_Lag1': [future_values[company]['Revenue_Lag1']],
        'Revenue_Lag2': [future_values[company]['Revenue_Lag2']],
        'Market Cap(in B USD)': [future_values[company]['Market Cap(in B USD)']],
        'Gross Profit': [future_values[company]['Gross Profit']],
        'Net Income': [future_values[company]['Net Income']],
        'EBITDA': [future_values[company]['EBITDA']],
        'Inflation Rate(in US)': [future_values[company]['Inflation Rate(in US)']],
    }

    # Convert future features to DataFrame
    future_data = pd.DataFrame(future_features)

    # Scale the future features using the same scaler
    future_data_scaled = scaler.transform(future_data)

    # Predict future revenue
    future_predicted_revenue = model.predict(future_data_scaled)

    # Print or use future predicted revenue
    print(f"Predicted revenue for {company} for the next year: {future_predicted_revenue}")


# In[ ]:




