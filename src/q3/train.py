import keras
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import tensorflow as tf
from data import *


#creating the training data set, where y lags by 1
X = time_series['POWER'][:-1]
y = time_series['POWER'].shift(-1)[:-1].values
X = X.values.reshape(-1, 1)
last = time_series['POWER'].shift(-1).iloc[-2]
last_value = np.array([last])


def prediction(model, filename):
    export = forecast.copy()
    predictions = []
    current_input = last_value
    for _ in range(len(export)):  # for each hour in the next month
        next_prediction = model.predict(current_input)
        predictions.append(next_prediction.item())
        current_input = [next_prediction]
    export['FORECAST'] = predictions
    export.to_csv(f"export/ForecastTemplate3-{filename}.csv", index=False)
    return export

def prediction_rnn(model, filename):
    export = forecast.copy()
    predictions = []
    current_input = last_value.reshape(1, 1, 1)
    for _ in range(len(export)):  # for each hour in the next month
        next_prediction = model.predict(current_input)
        predictions.append(next_prediction.item())
        current_input = next_prediction.reshape(1, 1, 1)
    export['FORECAST'] = predictions
    export.to_csv(f"export/ForecastTemplate3-{filename}.csv", index=False)
    return export

def calculate_rmse(results):
    return np.sqrt(mean_squared_error(solution["POWER"], results["FORECAST"]))


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
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=30, batch_size=16)

    results = prediction(model, "ANN")
    rmse = calculate_rmse(results)
    print("ANN RMSE:", rmse)

# Recurrent Neural Network

def rnn():
    X_train = X.reshape(X.shape[0], 1, X.shape[1])

    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=(1, 1)),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y, epochs=30, batch_size=16)

    results = prediction_rnn(model, "RNN")
    rmse = calculate_rmse(results)
    print(f"RNN RMSE: {rmse}")


# Prediction

if __name__ == "__main__":
    rnn()
