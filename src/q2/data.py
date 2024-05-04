import pandas as pd
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[2]
train_path = root.joinpath("data", "TrainData.csv")
template_path = root.joinpath("data", "ForecastTemplate.csv")
solution_path = root.joinpath("data", "Solution.csv")

train = pd.read_csv(train_path)

# calculate direction based on u and v components
def calculate_wind_direction(u10, v10):
    return (90 - np.arctan2(v10, u10) * 180 / np.pi) % 360


train['WindDirection'] = calculate_wind_direction(train['U10'], train['V10'])
train = train[['TIMESTAMP', 'POWER', 'WS10', 'WindDirection']]
train.rename(columns={'WS10': 'WindSpeed'}, inplace=True)
