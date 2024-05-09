#imports
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import sigmoid
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Read data from CSV
data = pd.read_csv('TrainData.csv')


# Load data from Excel file
data = pd.read_csv('TrainData.csv')
forcase = pd.read_csv('WeatherForecastInput.csv')


# Separate input (X) and output (y) columns
data['arctan_10'] = np.arctan(data['V10'] / data['U10'])
print(data[['arctan_10']])
data['arctan_100'] = np.arctan(data['V100'] / data['U100'])
print(data[['arctan_10']])
X = data[['arctan_10', 'WS10','arctan_100','WS100']]
y = data['POWER']

# Define the model
model = Sequential()
model.add(Dense(3, input_dim=4, activation=sigmoid))  # Hidden layer with 3 nodes
model.add(Dense(1, activation=sigmoid))              # Output layer with 1 node

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=100, batch_size=10)  # Adjust epochs and batch_size as needed

# process the weatherforecast data
forcase['arctan_10'] = np.arctan(forcase['V10'] / forcase['U10'])
print(forcase[['arctan_10']])
forcase['arctan_100'] = np.arctan(forcase['V100'] / forcase['U100'])
print(forcase[['arctan_10']])
X_pred = forcase[['arctan_10', 'WS10','arctan_100','WS100']]

y_pred = model.predict (X_pred)

print(y_pred)
print("size of y_pred:", len(y_pred))


solution = pd.read_csv("Solution.csv")
dates = pd.to_datetime(solution["TIMESTAMP"], format="%Y%m%d %H:%M")
plt.figure(figsize=(12, 8))
plt.plot(dates, solution['POWER'], label='True Wind Power')
plt.plot(dates, y_pred, label=' Predicted Power of Multivariable Neural Network Model with 1 Hidden Layer')
# plt.plot(dates, y_pred, label='SVR Predicted Power')
plt.title(f"True Wind Power vs Multivariable ANN Model Prediction")
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Wind Power")
plt.legend()
plt.tight_layout()
plt.show()

RMSE = np.sqrt(mean_squared_error(solution["POWER"], y_pred))
# RMSE_lr = np.sqrt(mean_squared_error(solution["POWER"], y_pred_lr))
print("RMSE for Multivariable ANN =",RMSE)
# print("RMSE for lr=",RMSE_lr)
