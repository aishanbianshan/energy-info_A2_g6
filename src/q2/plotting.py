import matplotlib.pyplot as plt
from data import *
from MLR import *

plt.figure(figsize=(12, 8))

plt.plot(dates, forecast_y, label='True Wind Power')
plt.plot(dates, mlr_y_pred, label='MLR Predicted Power')
plt.plot(dates, lr_y_pred, label='LR Predicted Power')

plt.title(f"True Wind Power vs LR and MLR Predicted Power")
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Wind Power")
plt.legend()
plt.tight_layout()

plt.show()
