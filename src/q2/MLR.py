from math import sqrt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from data import *

lr_X = train[['WindSpeed']]
mlr_X = train[['WindSpeed', 'WindDirection']]
y = train['POWER']

lr = LinearRegression()
mlr = LinearRegression()
lr.fit(lr_X, y)
mlr.fit(mlr_X, y)

# coefficients and intercept
print("Coefficients:", mlr.coef_)
print("Intercept:", mlr.intercept_)

mlr_forecast_X = forecast[['WindSpeed', 'WindDirection']]
lr_forecast_X = forecast[['WindSpeed']]
lr_y_pred = lr.predict(lr_forecast_X)
mlr_y_pred = mlr.predict(mlr_forecast_X)
forecast_y = solution

lr_rmse = sqrt(mean_squared_error(forecast_y, lr_y_pred))
mlr_rmse = sqrt(mean_squared_error(forecast_y, mlr_y_pred))
print("LR RMSE:", lr_rmse)
print("MLR RMSE:", mlr_rmse)


export = template.copy()
export['FORECAST'] = mlr_y_pred
export.to_csv(f"export/ForecastTemplate2.csv", index=False)
