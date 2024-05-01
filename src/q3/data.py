import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data_src = root.joinpath("data", "TrainData.csv")
template_src = root.joinpath("data", "ForecastTemplate.csv")
solution_src = root.joinpath("data", "Solution.csv")

time_series = pd.read_csv(data_src, usecols=['TIMESTAMP', 'POWER'])
forecast = pd.read_csv(template_src, usecols=['TIMESTAMP'])
solution = pd.read_csv(solution_src, usecols=['POWER'])

time_series['TIME'] = time_series.index
forecast['TIME'] = forecast.index
solution['TIME'] = solution.index
