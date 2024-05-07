# from q1

def plot_training(predictions, model_name):
    print(
        "Root Mean Squared Error(RMSE): ",
        sqrt(mean_squared_error(solution_vector, predictions)),
    )

    # plot the data
    plt.figure(figsize=(6, 4.35))
    plt.scatter(X, y, color="black", facecolor="none", edgecolor="black", label="Training data")
    plt.scatter(test_set, predictions, color="red", label="Predictions")
    plt.title(f"Training data - {model_name}")
    plt.xlabel("Wind Speed")
    plt.ylabel("Wind Power")
    plt.legend()

    path = root.joinpath("fig", f"q1-{model_name}-training.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")



def plot_forecast(predictions, model_name):
    # line plot of the predictied power for each hour
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_dates, solution_vector, label="Actual Power")
    plt.plot(forecast_dates, predictions, label="Predicted Power")
    plt.title(f"Forecasted Power vs. Actual Power - {model_name}")
    plt.xticks(rotation=45)
    plt.xlabel("Time")
    plt.ylabel("Wind Power")
    plt.legend()
    plt.tight_layout()

    path = root.joinpath("fig", f"q2-{model_name}-forecast.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
