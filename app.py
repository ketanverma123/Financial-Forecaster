from flask import Flask, request, render_template
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

def load_and_process_data(file_path=None, data_frame=None):
    if file_path:
        data = pd.read_csv(file_path)
    elif data_frame is not None:
        data = data_frame
    else:
        return None, None

    data = data.rename(columns={'Company ': 'Company'})

    required_columns = [
        'Year', 'Quarter', 'Company', 'Revenue', 'Gross Profit', 'Net Income',
        'Earning Per Share', 'EBITDA', 'Share Holder Equity', 'Current Ratio',
        'Debt/Equity Ratio', 'ROE', 'ROA'
    ]
    try:
        data_selected = data[required_columns]
    except KeyError:
        return None, None

    imputer = SimpleImputer(strategy='mean')
    data_imputed_numeric = pd.DataFrame(
        imputer.fit_transform(data_selected.iloc[:, 3:]),
        columns=data_selected.columns[3:]
    )
    data_imputed = pd.concat([data_selected[['Year', 'Quarter', 'Company']], data_imputed_numeric], axis=1)

    label_encoder = LabelEncoder()
    data_imputed['CompanyEncoded'] = label_encoder.fit_transform(data_imputed['Company'])

    return data_imputed, label_encoder

default_data_imputed, default_label_encoder = load_and_process_data(file_path=r'C:\Users\kavin\Downloads\Quarterfinancial.csv')

def get_next_quarter(year, quarter, increment=1):
    new_quarter = quarter + increment
    year_increment = (new_quarter - 1) // 4
    new_quarter = ((new_quarter - 1) % 4) + 1
    return int(year + year_increment), int(new_quarter)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    num_quarters = int(request.form.get('num_quarters', 2))

    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_extension = os.path.splitext(filename)[1].lower()

        try:
            if file_extension == '.csv':
                # Read CSV file with default encoding
                user_data = pd.read_csv(file)
            elif file_extension in ['.xls', '.xlsx']:
                # Read Excel file
                user_data = pd.read_excel(file)
            else:
                return render_template('results.html', predictions=None, error="Unsupported file format. Please upload a CSV or Excel file.")

        except Exception as e:
            return render_template('results.html', predictions=None, error=f"Failed to read the uploaded file: {e}")

        data_imputed, label_encoder = load_and_process_data(data_frame=user_data)

        if data_imputed is None:
            return render_template('results.html', predictions=None, error="Uploaded file is invalid or missing required columns.")

        companies_to_predict = [company] if company else data_imputed['Company'].unique()

        predictions_list = []

        X = data_imputed[['Year', 'Quarter', 'CompanyEncoded']]
        y = data_imputed.drop(columns=['Year', 'Quarter', 'Company', 'CompanyEncoded'])

        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        model.fit(X, y)

        for comp in companies_to_predict:
            try:
                company_encoded = label_encoder.transform([comp])[0]
            except ValueError:
                # Company not found in user data
                continue  # Skip to next company

            recent_records = data_imputed[data_imputed['CompanyEncoded'] == company_encoded]
            if recent_records.empty:
                continue  # Skip to next company

            recent_record = recent_records.sort_values(by=['Year', 'Quarter']).iloc[-1]
            predictions = []

            for i in range(1, num_quarters + 1):
                next_year, next_quarter = get_next_quarter(
                    recent_record['Year'], recent_record['Quarter'], increment=i
                )
                input_data = pd.DataFrame([[next_year, next_quarter, company_encoded]], columns=['Year', 'Quarter', 'CompanyEncoded'])
                prediction = model.predict(input_data)
                prediction_df = pd.DataFrame(prediction, columns=y.columns)
                prediction_df['Year'] = next_year
                prediction_df['Quarter'] = next_quarter
                prediction_df['Company'] = comp
                predictions.append(prediction_df)

            if predictions:
                predictions_list.append(pd.concat(predictions))

    else:
        data_imputed = default_data_imputed
        label_encoder = default_label_encoder

        companies_to_predict = [company] if company else data_imputed['Company'].unique()

        predictions_list = []

        X = data_imputed[['Year', 'Quarter', 'CompanyEncoded']]
        y = data_imputed.drop(columns=['Year', 'Quarter', 'Company', 'CompanyEncoded'])

        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        model.fit(X, y)

        for comp in companies_to_predict:
            try:
                company_encoded = label_encoder.transform([comp])[0]
            except ValueError:
                continue  

            recent_records = data_imputed[data_imputed['CompanyEncoded'] == company_encoded]
            if recent_records.empty:
                continue  

            recent_record = recent_records.sort_values(by=['Year', 'Quarter']).iloc[-1]
            predictions = []

            for i in range(1, num_quarters + 1):
                next_year, next_quarter = get_next_quarter(
                    recent_record['Year'], recent_record['Quarter'], increment=i
                )
                input_data = pd.DataFrame([[next_year, next_quarter, company_encoded]], columns=['Year', 'Quarter', 'CompanyEncoded'])
                prediction = model.predict(input_data)
                prediction_df = pd.DataFrame(prediction, columns=y.columns)
                prediction_df['Year'] = next_year
                prediction_df['Quarter'] = next_quarter
                prediction_df['Company'] = comp
                predictions.append(prediction_df)

            if predictions:
                predictions_list.append(pd.concat(predictions))

    if not predictions_list:
        return render_template('results.html', predictions=None, error="No predictions could be made. Please check your inputs.")

    final_predictions = pd.concat(predictions_list, ignore_index=True)

    return render_template('results.html', predictions=final_predictions.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)

