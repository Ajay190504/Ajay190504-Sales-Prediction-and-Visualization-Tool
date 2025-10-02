
# Sales Prediction and Visualization Tool

## How to Run Locally

1. Clone the repo:
   ```
   git clone https://github.com/<your-username>/Sales-Prediction-and-Visualization-Tool.git
   cd Sales-Prediction-and-Visualization-Tool
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Using the App

1. Upload your sales data (CSV or Excel). Ensure it has columns: `Date` and `Sales`.
2. Choose a visualization for your uploaded data (Line, Bar, Histogram, Scatter).
3. Pick a forecasting model: Prophet or ARIMA.
4. Specify forecast horizon (months).
5. View forecast results and choose visualization (Line, Bar, Table, or Confidence Interval Plot).
6. Optionally, try the included `sample_sales_data.csv` to test instantly.
