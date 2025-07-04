import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# Step 1: Read Excel and clean
df = pd.read_excel("budg_exm.xlsx", parse_dates=["Month"])
df.columns = df.columns.str.strip()
df['Month'] = pd.to_datetime(df['Month']).dt.to_period('M').dt.to_timestamp()

# Step 2: Melt to long format
df_long = df.melt(id_vars=["Month"], var_name="Department", value_name="Expense")
df_long['Expense'] = df_long['Expense'].astype(float)

# Step 3: Forecast
all_forecasts = []

for dept in df_long['Department'].unique():
    dept_df = df_long[df_long['Department'] == dept][['Month', 'Expense']].rename(columns={'Month': 'ds', 'Expense': 'y'})
    m = Prophet()
    m.fit(dept_df)
    

    future = m.make_future_dataframe(periods=13, freq='M')
    forecast = m.predict(future)

    # Filter forecast where 'ds' > max real data
    last_actual = dept_df['ds'].max()
    
    # Convert to Period Month for comparison
    forecast['ds_period'] = forecast['ds'].dt.to_period('M')
    last_period = last_actual.to_period('M')

    # Filter only future months
    forecast_trim = forecast[forecast['ds_period'] > last_period][['ds', 'yhat']]
    forecast_trim['Department'] = dept

    all_forecasts.append(forecast_trim)

# Step 4: Combine all forecasts
forecast_df = pd.concat(all_forecasts)

# Step 5: Pivot to wide
pivot_forecast = forecast_df.pivot(index='ds', columns='Department', values='yhat').reset_index()
pivot_forecast['ds'] = pivot_forecast['ds'].dt.strftime('%b-%Y')
pivot_forecast.rename(columns={'ds': 'Month'}, inplace=True)

# Step 6: Format original data
df['Month'] = pd.to_datetime(df['Month']).dt.strftime('%b-%Y')

# Step 7: Combine actual + forecast
final_df = pd.concat([df, pivot_forecast], ignore_index=True)
final_df.to_excel("department_expense_forecast.xlsx", index=False)


