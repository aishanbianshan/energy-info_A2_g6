import keras
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import tensorflow as tf
from data import *


windowsize = 1

def create_training(dataset, windowsize):
    rows = len(dataset) - windowsize
    x = np.zeros((rows, windowsize))
    y = np.zeros(rows)
    for i in range(rows):
        for j in range(windowsize):
            x[i][j] = dataset[i + j]
        y[i] = dataset[i + windowsize]
    return x, y


#creating the training data set, where y lags by 1
power = time_series['POWER'].to_numpy()
X, y = create_training(power, windowsize)
actual = solution['POWER'].to_numpy()
forecast_input, _ = create_training(actual, windowsize)

def prediction(model, filename):
    export = forecast.copy()[:-1]
    predicted = model.predict(forecast_input)
    export['FORECAST'] = predicted
    export.to_csv(f"export/ForecastTemplate3-{filename}.csv", index=False)
    return export

def calculate_rmse(results):
    return np.sqrt(mean_squared_error(actual[windowsize:], results["FORECAST"]))


# Linear Regression
def linear():
    model = LinearRegression()
    model.fit(X, y)
    results = prediction(model, "LR")
    rmse = calculate_rmse(results)
    print(f"LR RMSE: {rmse}")


# Support Vector Regression
def svr():
    model = SVR(kernel='rbf', C=0.01, gamma=0.1, epsilon=.1)
    model.fit(X, y)
    results = prediction(model, "SVR")
    rmse = calculate_rmse(results)
    print(f"SVR RMSE: {rmse}")

# Artificial Neural Network
def ann():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu", input_shape=(1,)),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=20, batch_size=16)

    results = prediction(model, "ann")
    rmse = calculate_rmse(results)
    print("ANN RMSE:", rmse)

# Recurrent Neural Network

def rnn():
    X_train = X.reshape(X.shape[0], 1, X.shape[1])

    model = keras.Sequential([
        keras.layers.LSTM(50, input_shape=(1, 1)),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y, epochs=20, batch_size=16)

    results = prediction(model, "RNN")
    rmse = calculate_rmse(results)
    print(f"RNN RMSE: {rmse}")


# Prediction

if __name__ == "__main__":
    ann()
