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

# Extract features (X) and target variable (y)
X = data['POWER'].iloc[:15359].values.reshape(-1, 1)  # Reshape to 2D array for sklearn
y = data['POWER'].iloc[1:15360].values

svr = SVR(kernel='rbf')  # RBF kernel is commonly used for SVR
svr.fit(X, y)

# Standardize the input for prediction
scaler = StandardScaler()
X_pred = scaler.fit_transform(data['POWER'].iloc[15360:].values.reshape(-1, 1))

# Predict y for the input X
y_pred = svr.predict(X_pred)

# Print the predicted values
# print("Predicted values:", y_pred)

# learn regression worked!!
# Create and fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Extract features for prediction (X_pred)
X_pred_lr = data['POWER'].iloc[15360:].values.reshape(-1, 1)  # Reshape to 2D array for sklearn

# Predict y for X_pred
y_pred_lr = model.predict(X_pred_lr)

# print("Predicted y:", y_pred)
# print("number of prediction in y:", len(y_pred))

solution = pd.read_csv("Solution.csv")
dates = pd.to_datetime(solution["TIMESTAMP"], format="%Y%m%d %H:%M")
plt.figure(figsize=(12, 8))
plt.plot(dates, solution['POWER'], label='True Wind Power')
plt.plot(dates, y_pred_lr, label='LR Predicted Power')
plt.plot(dates, y_pred, label='SVR Predicted Power')
plt.title(f"True Wind Power vs LR and SVR Predicted Power")
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Wind Power")
plt.legend()
plt.tight_layout()
plt.show()


RMSE = np.sqrt(mean_squared_error(solution["POWER"], y_pred))
RMSE_lr = np.sqrt(mean_squared_error(solution["POWER"], y_pred_lr))
print("RMSE for svr=",RMSE)
print("RMSE for lr=",RMSE_lr)
