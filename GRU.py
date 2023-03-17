import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Historic_Crypto import HistoricalData
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
import datetime
import sys

cryptocurrency = sys.argv[1]
eth = HistoricalData(f"{cryptocurrency}-EUR", 60*60*24, '2016-01-01-00-00', '2023-03-24-00-00').retrieve_data()



# Remove any missing values or outliers
eth = eth.dropna()
eth = eth[eth["close"] > 0]

# Transform the data into a suitable format for analysis
X = eth[["open", "high", "low", "volume"]]
y = eth["close"]


# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define the model
model = Sequential()
model.add(GRU(128, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(64))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=adam)

# Reshape data for GRU model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Train the model
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))


# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error of the model
mse = mean_squared_error(y_test, y_pred)

# Calculate the root mean squared error of the model
rmse = np.sqrt(mse)

# Calculate the mean absolute error of the model
mae = mean_absolute_error(y_test, y_pred)

# Print the evaluation metrics
print('Mean Squared Error (MSE): ', mse)
print('Root Mean Squared Error (RMSE): ', rmse)
print('Mean Absolute Error (MAE): ', mae)

plt.plot(range(len(y_train)), y_train, color='orange', label='Training Data') # Plot the training data
plt.plot(range(len(y_train), len(y_train)+len(y_test)), y_test, color='blue', label='Testing Data') # Plot the testing data

plt.title('Training and Testing Data Split')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot the training loss and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(0, len(history.history['loss']), 20))
plt.legend()
plt.show()

plt.title('Actual Price vs Predicted Price')
plt.plot(eth.index[-len(y_test):], y_test, label='Actual Price')
plt.plot(eth.index[-len(y_test):], y_pred, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Make predictions on future prices
today = datetime.date.today()
forecast = model.predict(np.reshape(scaler.transform(X[-30:]), (30, 4, 1)))

# Plot the actual and predicted prices for the next 14 days
plt.plot(eth.index[-30:], eth['close'][-30:], label='Actual Price')
plt.axvline(x=today, color='red', linestyle='--', label='Prediction Start')
plt.plot(pd.date_range(start=today, periods=14), forecast[-14:], label='Predicted Price')
plt.title('Actual Price and Future Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=30)
plt.savefig(f'{cryptocurrency}_plot.png', dpi=300)
plt.show()