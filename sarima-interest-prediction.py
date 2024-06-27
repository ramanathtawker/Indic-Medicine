import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools

# Step 1: Load and prepare the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# Step 2: Visualize the interest amount time series
def plot_interest_time_series(series):
    plt.figure(figsize=(12, 6))
    plt.plot(series)
    plt.title('Interest Amount Time Series')
    plt.xlabel('Date')
    plt.ylabel('Interest Amount')
    plt.show()

# Step 3: Check stationarity and seasonality
def check_stationarity_seasonality(series):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, ax=ax1, title='Autocorrelation Function (ACF)')
    plot_pacf(series, ax=ax2, title='Partial Autocorrelation Function (PACF)')
    plt.tight_layout()
    plt.show()

# Step 4: Manual grid search for SARIMA parameters
def manual_grid_search(series):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    best_aic = np.inf
    best_params = None
    best_seasonal_params = None
    
    for param in pdq:
        for seasonal_param in seasonal_pdq:
            try:
                model = SARIMAX(series,
                                order=param,
                                seasonal_order=seasonal_param,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = param
                    best_seasonal_params = seasonal_param
                print(f'SARIMA{param}x{seasonal_param} - AIC:{results.aic}')
            except:
                continue
    
    print(f'Best SARIMA parameters: SARIMA{best_params}x{best_seasonal_params}')
    return best_params, best_seasonal_params

# Step 5: Fit SARIMA model
def fit_sarima_model(series, order, seasonal_order):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    results = model.fit()
    print(results.summary())
    return results

# Step 6: Make predictions and evaluate model
def predict_and_evaluate(results, series, test_size=12):
    # Split data into train and test
    train = series[:-test_size]
    test = series[-test_size:]
    
    # Forecast
    forecast = results.get_forecast(steps=test_size)
    forecast_mean = forecast.predicted_mean
    
    # Calculate error metrics
    mse = mean_squared_error(test, forecast_mean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, forecast_mean)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Test Data')
    plt.plot(test.index, forecast_mean, label='Predicted')
    plt.fill_between(test.index,
                     forecast.conf_int()['lower Interest Amount'],
                     forecast.conf_int()['upper Interest Amount'],
                     color='k', alpha=0.1)
    plt.title('SARIMA Model: Actual vs Predicted Interest Amounts')
    plt.legend()
    plt.show()
    
    return forecast_mean

# Main function to run the SARIMA model development process
def predict_interest_amount(file_path):
    # Load data
    df = load_data(file_path)
    series = df['Interest Amount']  # Assuming the column is named 'Interest Amount'
    
    # Visualize time series
    plot_interest_time_series(series)
    
    # Check stationarity and seasonality
    check_stationarity_seasonality(series)
    
    # Find best parameters
    order, seasonal_order = manual_grid_search(series)
    
    # Fit model
    results = fit_sarima_model(series, order, seasonal_order)
    
    # Make predictions and evaluate
    forecast = predict_and_evaluate(results, series)
    
    # Future predictions
    future_steps = 12  # Predict next 12 periods
    future_forecast = results.get_forecast(steps=future_steps)
    print("\nForecast for the next 12 periods:")
    print(future_forecast.predicted_mean)
    
    # Plot future forecast
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series, label='Historical Data')
    plt.plot(pd.date_range(start=series.index[-1], periods=future_steps+1, freq='M')[1:],
             future_forecast.predicted_mean, label='Future Forecast')
    plt.fill_between(pd.date_range(start=series.index[-1], periods=future_steps+1, freq='M')[1:],
                     future_forecast.conf_int()['lower Interest Amount'],
                     future_forecast.conf_int()['upper Interest Amount'],
                     color='k', alpha=0.1)
    plt.title('SARIMA Model: Future Interest Amount Forecast')
    plt.legend()
    plt.show()

# Usage
file_path = 'your_interest_data.csv'
predict_interest_amount(file_path)
