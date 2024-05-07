import matplotlib.pyplot as plt
from data import *
from MLR import *

plt.figure(figsize=(12, 8))

plt.plot(dates, forecast_y, label='True Wind Power', color='green',  alpha=0.8)
plt.plot(dates, mlr_y_pred, label='MLR Predicted Power', color='navy', alpha=0.7)
plt.plot(dates, lr_y_pred, label='LR Predicted Power', color='deeppink', alpha=0.7)

plt.title(f"True Wind Power vs LR and MLR Predicted Power")
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Wind Power")
plt.legend()
plt.tight_layout()

plt.show()
