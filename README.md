VIEW IN CODE FOR BETTER FORMAT

This project is a web application that provides financial predictions for various companies based on historical financial data. The application uses a machine learning model to predict key financial metrics for a specified number of quarters. Users can select a company from a predefined list and input the number of future quarters for which they want predictions.
Features
User-Friendly Interface: Built with Flask and Bootstrap to ensure a responsive and user-friendly interface.
Company Selection: Users can select a company from a list of available companies.
Quarterly Predictions: The application provides financial predictions for multiple quarters based on historical data.
Visual Results: Predictions are displayed in a clean and organized table format for easy interpretation.

v3 Updates:
In this version, the tool has been updated with the following features:

New Financial Indicators: Now includes Market Cap and Inflation Rate (in US) as part of the financial forecasting.
Improved Performance Suggestions: The suggestion system has been enhanced to offer more targeted recommendations based on financial trends.
Interactive Unit Conversion: The ability to switch between Billions, Millions, and Hundreds of Thousands for financial metrics.
UI Improvements: Cleaner, more professional user interface with improved formatting for instructions and feedback.
Back Button Navigation: Users can now easily navigate back to the landing page from the results page.

How It Works
Data Preparation: The application uses historical financial data from a CSV file. The data is cleaned and preprocessed to handle missing values and encode categorical variables.
Model Training: A Random Forest model is trained to predict multiple financial metrics simultaneously.
Prediction: Users select a company and the number of quarters to predict. The model then provides predictions for the specified future quarters.
Visualization: The predictions are presented in a formatted table on the results page.
Technologies Used
Python
Flask
Pandas
Scikit-Learn
Bootstrap
Installation and Setup
Clone the repository:

Copy code
git clone https://github.com/yourusername/financial-predictions-web-app.git
cd financial-predictions-web-app
Create and activate a virtual environment:

Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required packages:

Copy code
pip install -r requirements.txt
Start the Flask application:

Copy code
python app.py
Open a web browser and navigate to http://127.0.0.1:5000/ to access the application.

financial-predictions-web-app/
│
├── app.py
├── data/
│   └── Quarterfinancial.csv
├── models/
│   ├── model.pkl
│   └── label_encoder.pkl
├── templates/
    ├── index.html
    └── results.html
