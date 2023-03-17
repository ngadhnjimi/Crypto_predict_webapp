import numpy as np
import pandas as pd 
import datetime
import matplotlib.pyplot as plt

from Historic_Crypto import HistoricalData
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys

cryptocurrency = sys.argv[1]

# Load dataset
eth = HistoricalData(f"{cryptocurrency}-EUR", 60*60*24, '2017-01-01-00-00', '2023-03-24-00-00').retrieve_data()


# Fit model
def train_arima(eth):
    series = eth['close']
    model = ARIMA(series, order=(2,1,0))
    model_fit = model.fit()
    # Summary of fit model
    # print(model_fit.summary())
     # Plot training and validation loss
    model_fit.plot_diagnostics()
    plt.show()
    return series
     

def predict_arima(series):
    X = series.values
    size = int(len(X) * 0.7)
    train, test = X[0:size], X[size:len(X)]

     # Plot the training and testing data
    plt.plot(eth.index[0:size], train, label='Training Data')
    plt.plot(eth.index[size:len(X)], test, label='Testing Data')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Training and Testing Data Split')
    plt.legend()
    plt.show()

    history = [x for x in train]
    predictions = list()
    dates = eth.index[size:len(X)]
    # Walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        preds = output[0]
        predictions.append(preds)
        actual = test[t]
        history.append(actual)
    # Evaluate forecasts
    mse = mean_squared_error(test,predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)

    # Print the evaluation metrics
    print('Mean Squared Error (MSE): ', mse)
    print('Root Mean Squared Error (RMSE): ', rmse)
    print('Mean Absolute Error (MAE): ', mae)
    plt.title('Actual Price vs Predicted Price')

    plt.plot(dates, test)
    plt.plot(dates, predictions, color='orange')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
     

def forecast_arima(eth):
    # Get the last 30 days of data
    day_pred = eth.tail(30)['close']

    # Fit the model on the last 30 days of data
    model = ARIMA(day_pred, order=(2, 1, 0))
    model_fit = model.fit()

    # Make predictions for the next 14 days
    forecast = model_fit.forecast(steps=14)
    return forecast

# Train ARIMA model
series = train_arima(eth)

# Make predictions on test data
predict_arima(series)

# Get the predicted prices for the next 14 days
forecast = forecast_arima(eth)

# Plot the actual and predicted prices
today = datetime.date.today()
day_pred = eth.tail(30)['close']
plt.plot(day_pred.index, day_pred.values, label='Actual Price')
plt.axvline(x=today, color='red', linestyle='--', label='Prediction Start')
plt.plot(pd.date_range(start=today, periods=14), forecast, label='Predicted Price')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual and Future Price')
plt.legend()
plt.xticks(rotation=30)
plt.savefig(f'{cryptocurrency}_plot.png', dpi=300)
plt.show()