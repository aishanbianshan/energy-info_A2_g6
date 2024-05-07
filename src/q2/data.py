import pandas as pd
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[2]
train_path = root.joinpath("data", "TrainData.csv")
forecast_path = root.joinpath("data", "WeatherForecastInput.csv")
template_path = root.joinpath("data", "ForecastTemplate.csv")
solution_path = root.joinpath("data", "Solution.csv")

train = pd.read_csv(train_path)
forecast = pd.read_csv(forecast_path)
solution = pd.read_csv(solution_path)


# calculate direction based on u and v components
def calculate_wind_direction(u10, v10):
    return (np.pi/2 - np.arctan2(v10, u10)) % (2*np.pi)


train['WindDirection'] = calculate_wind_direction(train['U10'], train['V10'])
forecast['WindDirection'] = calculate_wind_direction(forecast['U10'], forecast['V10'])
train = train[['TIMESTAMP', 'POWER', 'WS10', 'WindDirection']]
forecast = forecast[['TIMESTAMP', 'WS10', 'WindDirection']]
train.rename(columns={'WS10': 'WindSpeed'}, inplace=True)
forecast.rename(columns={'WS10': 'WindSpeed'}, inplace=True)
dates = pd.to_datetime(solution["TIMESTAMP"], format="%Y%m%d %H:%M")
solution = solution['POWER']

