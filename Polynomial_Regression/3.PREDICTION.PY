import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

DATA_DIR = "country_wise_data"
MAIN_OUTPUT_FOLDER = "Future Prediction Result"
os.makedirs(MAIN_OUTPUT_FOLDER, exist_ok=True)

for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    country_name = file.replace(".csv", "")
    country_df = pd.read_csv(os.path.join(DATA_DIR, file))

    required_cols = ["Year", "Yield", "Area_Harvested", "Item"]
    if not all(col in country_df.columns for col in required_cols):
        print(f"Skipping {country_name} (missing columns)")
        continue

    # ---- Country folder ----
    country_folder = os.path.join(
        MAIN_OUTPUT_FOLDER, f"{country_name}_future_graphs"
    )
    os.makedirs(country_folder, exist_ok=True)

    fruits = country_df["Item"].unique()

    for fruit in fruits:
        fruit_df = country_df[country_df["Item"] == fruit].sort_values("Year")

        if fruit_df.shape[0] < 5:
            continue

        # ===============================
        # POLYNOMIAL SETUP
        # ===============================
        poly = PolynomialFeatures(degree=2, include_bias=False)

        # ===============================
        # TRAIN YIELD MODEL (Polynomial, multivariate)
        # ===============================
        X = fruit_df[["Year", "Area_Harvested"]]
        Y = fruit_df["Yield"]

        X_poly = poly.fit_transform(X)

        yield_model = LinearRegression()
        yield_model.fit(X_poly, Y)

        # Historical fitted values
        y_pred_all = yield_model.predict(X_poly)

        # ===============================
        # PREDICT FUTURE AREA_HARVESTED (Polynomial)
        # ===============================
        future_years = [2026, 2027, 2028, 2029, 2030]

        X_area = fruit_df[["Year"]]
        Y_area = fruit_df["Area_Harvested"]

        poly_area = PolynomialFeatures(degree=2, include_bias=False)

        X_area_poly = poly_area.fit_transform(X_area)
        future_years_poly = poly_area.transform(
            pd.DataFrame({"Year": future_years})
        )

        area_model = LinearRegression()
        area_model.fit(X_area_poly, Y_area)

        future_area_pred = area_model.predict(future_years_poly)

        # ===============================
        # PREDICT FUTURE YIELD
        # ===============================
        future_df = pd.DataFrame({
            "Year": future_years,
            "Area_Harvested": future_area_pred
        })

        future_df_poly = poly.transform(future_df)
        future_yield_pred = yield_model.predict(future_df_poly)

        # ===============================
        # SAVE FUTURE RESULTS
        # ===============================
        future_results = pd.DataFrame({
            "Year": future_years,
            "Predicted_Area_Harvested": future_area_pred,
            "Predicted_Yield": future_yield_pred
        })

        future_results.to_csv(
            os.path.join(country_folder, f"{fruit}_2026_2030.csv"),
            index=False
        )

        # ===============================
        # PLOT
        # ===============================
        plt.figure(figsize=(10, 6))

        plt.scatter(
            fruit_df["Year"],
            fruit_df["Yield"],
            label="Actual Yield",
            s=20
        )

        plt.plot(
            fruit_df["Year"],
            y_pred_all,
            label="Polynomial Trend (Historical)",
            linewidth=2
        )

        plt.scatter(
            future_years,
            future_yield_pred,
            label="Forecast (2026–2030)",
            s=50
        )

        plt.title(f"{country_name} - {fruit}")
        plt.xlabel("Year")
        plt.ylabel("Yield")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(country_folder, f"{fruit}.png"))
        plt.close()

    print(f"✅ Done: {country_name} → saved in {country_folder}")
