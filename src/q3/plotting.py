import pandas as pd
import matplotlib.pyplot as plt

windowsize = 5

filepath = "export/"

ann = pd.read_csv(filepath+"ForecastTemplate3-ANN.csv")['FORECAST']
lr = pd.read_csv(filepath+"ForecastTemplate3-LR.csv")['FORECAST']
svr = pd.read_csv(filepath+"ForecastTemplate3-SVR.csv")['FORECAST']
rnn = pd.read_csv(filepath+"ForecastTemplate3-RNN.csv")['FORECAST']
solution = pd.read_csv(filepath+"Solution.csv")

dates = pd.to_datetime(solution["TIMESTAMP"], format="%Y%m%d %H:%M")

def lr_svr_plot():
    plt.figure(figsize=(12, 8))

    plt.plot(dates, solution['POWER'], label='True Wind Power')
    plt.plot(dates, lr, label='LR Predicted Power')
    plt.plot(dates, svr, label='SVR Predicted Power')

    plt.title(f"True Wind Power vs LR and SVR Predicted Power")
    plt.xticks(rotation=45)
    plt.xlabel("Time")
    plt.ylabel("Wind Power")
    plt.legend()
    plt.tight_layout()

    plt.show()

def nn_plot():
    dates_reduced = dates[:-windowsize]
    plt.figure(figsize=(12, 8))

    plt.plot(dates_reduced, solution['POWER'][:-windowsize], label='True Wind Power')
    plt.plot(dates_reduced, ann[:-windowsize], label='ANN Predicted Power')
    plt.plot(dates_reduced, rnn, label='RNN Predicted Power')

    plt.title(f"True Wind Power vs ANN and RNN Predicted Power")
    plt.xticks(rotation=45)
    plt.xlabel("Time")
    plt.ylabel("Wind Power")
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    nn_plot()
