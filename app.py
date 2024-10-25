from flask import Flask, request, render_template
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Function to load and process data
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


default_data_imputed, default_label_encoder = load_and_process_data(file_path='Quarterfinancial.csv')


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

    # Check if a file was uploaded
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_extension = os.path.splitext(filename)[1].lower()

        try:
            if file_extension == '.csv':
                user_data = pd.read_csv(file)
            elif file_extension in ['.xls', '.xlsx']:
                user_data = pd.read_excel(file)
            else:
                return render_template('results.html', predictions=None, error="Unsupported file format. Please upload a CSV or Excel file.")
        except Exception as e:
            return render_template('results.html', predictions=None, error=f"Failed to read the uploaded file: {e}")

        data_imputed, label_encoder = load_and_process_data(data_frame=user_data)
        if data_imputed is None:
            return render_template('results.html', predictions=None, error="Uploaded file is invalid or missing required columns.")

    else:
        data_imputed = default_data_imputed
        label_encoder = default_label_encoder

    companies_to_predict = [company] if company else data_imputed['Company'].unique()
    predictions_list = []

    X = data_imputed[['Year', 'Quarter', 'CompanyEncoded']]
    y = data_imputed.drop(columns=['Year', 'Quarter', 'Company', 'CompanyEncoded'])

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X, y)

    trend_feedback = {}
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
            company_predictions = pd.concat(predictions)
            predictions_list.append(company_predictions)

            # Generate feedback based on trends in key metrics
            revenue_trend = "increasing" if company_predictions['Revenue'].pct_change().mean() > 0 else "decreasing"
            net_income_trend = "profitable" if company_predictions['Net Income'].pct_change().mean() > 0 else "less profitable"
            debt_trend = "increasing reliance on debt" if company_predictions['Debt/Equity Ratio'].pct_change().mean() > 0 else "stable debt level"
            roe_trend = "efficient resource use" if company_predictions['ROE'].pct_change().mean() > 0 else "potential resource optimization needed"
            
            # New insights based on the predictions
            gross_profit_trend = "increasing" if company_predictions['Gross Profit'].pct_change().mean() > 0 else "decreasing"
            cost_optimization_needed = "Consider cost optimization strategies to reduce expenses." if gross_profit_trend == "increasing" and net_income_trend == "less profitable" else ""
            expand_market_strategy = "Consider expanding market reach or product offerings to boost revenue." if revenue_trend == "decreasing" else ""
            debt_management_strategy = "Consider restructuring debt or leveraging internal funds to reduce reliance on external debt." if debt_trend == "increasing reliance on debt" else ""

            trend_feedback[comp] = {
                "Revenue": revenue_trend,
                "NetIncome": net_income_trend,
                "DebtRatio": debt_trend,
                "ROE": roe_trend,
                "GrossProfit": gross_profit_trend,
                "Suggestions": {
                    "CostOptimization": cost_optimization_needed,
                    "MarketExpansion": expand_market_strategy,
                    "DebtManagement": debt_management_strategy
                }
            }

    if not predictions_list:
        return render_template('results.html', predictions=None, error="No predictions could be made. Please check your inputs.")

    final_predictions = pd.concat(predictions_list, ignore_index=True)

    return render_template('results.html', predictions=final_predictions.to_dict(orient='records'), trend_feedback=trend_feedback)

if __name__ == '__main__':
    app.run(debug=True)
