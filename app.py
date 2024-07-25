from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

app = Flask(__name__)

# Load the dataset
file_path = r'C:\Users\kavin\Downloads\Quarterfinancial.csv'
data = pd.read_csv(file_path)

# Correcting the column name
data = data.rename(columns={'Company ': 'Company'})

# Selecting the required columns
required_columns = [
    'Year', 'Quarter', 'Company', 'Revenue', 'Gross Profit', 'Net Income',
    'Earning Per Share', 'EBITDA', 'Share Holder Equity', 'Current Ratio',
    'Debt/Equity Ratio', 'ROE', 'ROA'
]
data_selected = data[required_columns]

# Handling missing values
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data_selected.iloc[:, 3:]), columns=data_selected.columns[3:])
data_imputed[['Year', 'Quarter', 'Company']] = data_selected[['Year', 'Quarter', 'Company']]

# Encoding categorical variable 'Company'
label_encoder = LabelEncoder()
data_imputed['Company'] = label_encoder.fit_transform(data_imputed['Company'])

# Splitting the data into features and target
X = data_imputed.drop(columns=['Revenue', 'Gross Profit', 'Net Income', 'Earning Per Share', 'EBITDA', 'Share Holder Equity', 'Current Ratio', 'Debt/Equity Ratio', 'ROE', 'ROA'])
y = data_imputed[['Revenue', 'Gross Profit', 'Net Income', 'Earning Per Share', 'EBITDA', 'Share Holder Equity', 'Current Ratio', 'Debt/Equity Ratio', 'ROE', 'ROA']]

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Function to get the next quarter
def get_next_quarter(year, quarter, increment=1):
    new_quarter = quarter + increment
    if new_quarter > 4:
        year += new_quarter // 4
        new_quarter = new_quarter % 4
    return year, new_quarter

# Function to predict the next quarters for a given company
def predict_next_quarters(company, num_quarters=2):
    company_encoded = label_encoder.transform([company])[0]
    
    # Get the most recent record for the company
    recent_record = data_imputed[data_imputed['Company'] == company_encoded].sort_values(by=['Year', 'Quarter']).iloc[-1]
    
    predictions = []
    for i in range(1, num_quarters + 1):
        next_year, next_quarter = get_next_quarter(recent_record['Year'], recent_record['Quarter'], increment=i)
        input_data = pd.DataFrame([[next_year, next_quarter, company_encoded]], columns=['Year', 'Quarter', 'Company'])
        prediction = model.predict(input_data)
        prediction_df = pd.DataFrame(prediction, columns=['Revenue', 'Gross Profit', 'Net Income', 'Earning Per Share', 'EBITDA', 'Share Holder Equity', 'Current Ratio', 'Debt/Equity Ratio', 'ROE', 'ROA'])
        prediction_df['Year'] = next_year
        prediction_df['Quarter'] = next_quarter
        prediction_df['Company'] = company
        predictions.append(prediction_df)
    
    return pd.concat(predictions)

@app.route('/')
def home():
    company_names = data['Company'].unique()
    return render_template('index.html', company_names=company_names)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    num_quarters = int(request.form['num_quarters'])
    
    # Make predictions
    predictions = predict_next_quarters(company, num_quarters=num_quarters)
    
    return render_template('results.html', predictions=predictions.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
