import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures   # <-- ADDED

DATA_DIR = "country_wise_data"
MAIN_OUTPUT_FOLDER = "Train-Test Result"
os.makedirs(MAIN_OUTPUT_FOLDER, exist_ok=True)

for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    country_name = file.replace(".csv", "")
    country_df = pd.read_csv(os.path.join(DATA_DIR, file))

    required_cols = ["Year", "Yield", "Area_Harvested", "Item"]
    if not all(col in country_df.columns for col in required_cols):
        print(f"⚠️ Skipping {country_name} (missing columns)")
        continue

    # Keep only 1996–2025
    country_df = country_df[
        (country_df["Year"] >= 1996) & (country_df["Year"] <= 2025)
    ]

    country_folder = os.path.join(
        MAIN_OUTPUT_FOLDER, f"{country_name}_multivar_graphs"
    )
    os.makedirs(country_folder, exist_ok=True)

    fruits = country_df["Item"].unique()

    for fruit in fruits:
        fruit_df = country_df[country_df["Item"] == fruit].sort_values("Year")

        if fruit_df.shape[0] < 10:
            continue

        # ---- Features and target ----
        X = fruit_df[["Year", "Area_Harvested"]]
        Y = fruit_df["Yield"]

        # ---- Time-based split ----
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, shuffle=False
        )

        # ========= POLYNOMIAL REGRESSION =========
        poly = PolynomialFeatures(degree=2, include_bias=False)

        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        X_poly_full = poly.transform(X)

        model = LinearRegression()
        model.fit(X_train_poly, Y_train)

        # ---- Predict ONLY on test set ----
        Y_test_pred = model.predict(X_test_poly)

        # ---- Evaluation ----
        mse = mean_squared_error(Y_test, Y_test_pred)
        r2 = r2_score(Y_test, Y_test_pred)

        # ---- Save test results ----
        test_result_df = X_test.copy()
        test_result_df["Actual_Yield"] = Y_test.values
        test_result_df["Predicted_Yield"] = Y_test_pred
        test_result_df["Residual"] = (
            test_result_df["Actual_Yield"] - test_result_df["Predicted_Yield"]
        )

        test_result_df.to_csv(
            os.path.join(country_folder, f"{fruit}_test_results.csv"),
            index=False
        )

        # ---- Polynomial trend line ----
        Y_trend_pred = model.predict(X_poly_full)

        # ---- Plot ----
        plt.figure(figsize=(10, 6))

        plt.scatter(X_train["Year"], Y_train, label="Train Actual", s=20)
        plt.scatter(X_test["Year"], Y_test, label="Test Actual", s=40)

        plt.plot(
            fruit_df["Year"],
            Y_trend_pred,
            label="Model Fit (Multivariate Polynomial Regression, degree=2)",
            linewidth=2
        )

        plt.title(
            f"{country_name} - {fruit} | Test MSE={mse:.2f} | Test R²={r2:.2f}"
        )
        plt.xlabel("Year")
        plt.ylabel("Yield")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(country_folder, f"{fruit}.png"))
        plt.close()

    print(f"✅ Done: {country_name} → saved in {country_folder}")
