from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from data import *

X = train[['WindSpeed', 'WindDirection']]
y = train['POWER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5410)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# rmse
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# TODO: predict forecast and compare with solution
model.predict()