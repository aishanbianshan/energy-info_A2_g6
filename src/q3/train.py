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
    X = time_series['TIME'].values.reshape(-1, 1)
    y = time_series['POWER']
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
    print("ANN RMSE:", rmse)
    export = fore_df[['TIMESTAMP', 'FORECAST']]
    export.to_csv("export/ForecastTemplate3-ANN.csv", index=False)


# Recurrent Neural Network
def create_sequences(df, target, window_size=5):
    sequences = []
    labels = []
    for i in range(len(df) - window_size):
        sequences.append(df.iloc[i:i+window_size][['hour', 'dayofweek', 'month']].values)
        labels.append(df.iloc[i + window_size][target])
    return np.array(sequences), np.array(labels)

def rnn():
    df = forecast.copy()
    ts = time_series.copy()
    ts['TIMESTAMP'] = pd.to_datetime(ts['TIMESTAMP'])
    ts['hour'] = ts['TIMESTAMP'].dt.hour
    ts['dayofweek'] = ts['TIMESTAMP'].dt.dayofweek
    ts['month'] = ts['TIMESTAMP'].dt.month

    window_size = 10
    X_train, y_train = create_sequences(ts, 'POWER', window_size)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Dense(1)
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=30, batch_size=16)

    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df['hour'] = df['TIMESTAMP'].dt.hour
    df['dayofweek'] = df['TIMESTAMP'].dt.dayofweek
    df['month'] = df['TIMESTAMP'].dt.month
    df['POWER'] = None

    X_pred, _ = create_sequences(df, "POWER", window_size)

    pred = model.predict(X_pred)
    df = fore_df.iloc[window_size:]
    sol = solution.iloc[window_size:]
    df["FORECAST"] = pred
    rmse = np.sqrt(((df["FORECAST"] - sol["POWER"]) ** 2).mean())
    print("RNN RMSE:", rmse)
    export = df[['TIMESTAMP', 'FORECAST']]
    export.to_csv("export/ForecastTemplate3-RNN.csv", index=False)

# Prediction

if __name__ == "__main__":
    linear()

