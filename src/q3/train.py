import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import tensorflow as tf
from data import *

X = time_series['TIME'].values.reshape(-1, 1)
y = time_series['POWER'].values
fore_df = forecast.copy()

# Linear Regression
def linear():
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(forecast['TIME'].values.reshape(-1, 1))
    fore_df["FORECAST"] = pred
    rmse = np.sqrt(((fore_df["FORECAST"] - solution["POWER"]) ** 2).mean())
    print("LR RMSE:", rmse)
    export = fore_df[['TIMESTAMP', 'FORECAST']]
    export.to_csv("export/ForecastTemplate3-LR.csv", index=False)

# Support Vector Regression
def svr():
    model = SVR(kernel='rbf', C=0.01, gamma=0.1, epsilon=.1)
    model.fit(X, y)
    pred = model.predict(forecast['TIME'].values.reshape(-1, 1))
    fore_df["FORECAST"] = pred
    rmse = np.sqrt(((fore_df["FORECAST"] - solution["POWER"]) ** 2).mean())
    print("SVR RMSE:", rmse)
    export = fore_df[['TIMESTAMP', 'FORECAST']]
    export.to_csv("export/ForecastTemplate3-SVR.csv", index=False)

# Artificial Neural Network
def ann():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu", input_shape=(1,)),
            tf.keras.layers.Dense(5, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=50, batch_size=16)

    pred = model.predict(forecast['TIME'].values.reshape(-1, 1))
    fore_df["FORECAST"] = pred
    rmse = np.sqrt(((fore_df["FORECAST"] - solution["POWER"]) ** 2).mean())
    print("SVR RMSE:", rmse)
    export = fore_df[['TIMESTAMP', 'FORECAST']]
    export.to_csv("export/ForecastTemplate3-ANN.csv", index=False)


# Recurrent Neural Network
def create_sequences(data, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:(i + window_size)])
        labels.append(data[i + window_size])
    return np.array(sequences), np.array(labels)

def rnn():
    window_size = 5
    rnn_X, rnn_y = create_sequences(time_series['POWER'].values, window_size)
    rnn_X = rnn_X.reshape((rnn_X.shape[0], rnn_X.shape[1], 1))
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(window_size, 1)),
            tf.keras.layers.Dense(1)
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(rnn_X, rnn_y, epochs=20, batch_size=16)

    X_pred, _ = create_sequences(fore_df['TIME'].values, window_size)
    X_pred = X_pred.reshape((X_pred.shape[0], X_pred.shape[1], 1))

    pred = model.predict(X_pred)
    print(pred)
    fore_df["FORECAST"] = pred
    rmse = np.sqrt(((fore_df["FORECAST"] - solution["POWER"]) ** 2).mean())
    print("SVR RMSE:", rmse)
    export = fore_df[['TIMESTAMP', 'FORECAST']]
    export.to_csv("export/ForecastTemplate3-RNN.csv", index=False)

# Prediction

if __name__ == "__main__":
    rnn()
