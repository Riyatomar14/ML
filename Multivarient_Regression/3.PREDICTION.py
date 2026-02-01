import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

DATA_DIR = "country_wise_data"
MAIN_OUTPUT_FOLDER = "Future Prediction Result"
os.makedirs(MAIN_OUTPUT_FOLDER, exist_ok=True)  # create main folder

for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    country_name = file.replace(".csv", "")
    country_df = pd.read_csv(os.path.join(DATA_DIR, file))

    required_cols = ["Year", "Yield", "Area_Harvested", "Item"]
    if not all(col in country_df.columns for col in required_cols):
        print(f"Skipping {country_name} (missing columns)")
        continue

    # Create ONE folder per country inside main folder
    country_folder = os.path.join(MAIN_OUTPUT_FOLDER, f"{country_name}_future_graphs")
    os.makedirs(country_folder, exist_ok=True)

    fruits = country_df["Item"].unique()

    for fruit in fruits:
        fruit_df = country_df[country_df["Item"] == fruit].sort_values("Year")

        if fruit_df.shape[0] < 5:
            continue

        # -------- Train Yield model (multivariate) --------
        X = fruit_df[["Year", "Area_Harvested"]]
        Y = fruit_df["Yield"]

        yield_model = LinearRegression()
        yield_model.fit(X, Y)

        # Predicted trend (historical)
        y_pred_all = yield_model.predict(X)

        # -------- Predict future Area_Harvested --------
        future_years = [2026, 2027, 2028, 2029, 2030]
        X_area = fruit_df[["Year"]]
        Y_area = fruit_df["Area_Harvested"]

        area_model = LinearRegression()
        area_model.fit(X_area, Y_area)

        future_area_pred = area_model.predict(pd.DataFrame({"Year": future_years}))

        # -------- Predict future Yield using predicted Area_Harvested --------
        future_df = pd.DataFrame({
            "Year": future_years,
            "Area_Harvested": future_area_pred
        })

        future_yield_pred = yield_model.predict(future_df)

        # -------- Save future predictions CSV --------
        future_results = pd.DataFrame({
            "Year": future_years,
            "Predicted_Area_Harvested": future_area_pred,
            "Predicted_Yield": future_yield_pred
        })

        future_results.to_csv(
            os.path.join(country_folder, f"{fruit}_2026_2030.csv"),
            index=False
        )

        # -------- Plot historical Yield and predicted Yield --------
        plt.figure(figsize=(10, 6))

        # Historical actual yield
        plt.scatter(
            fruit_df["Year"],
            fruit_df["Yield"],
            label="Actual Yield",
            s=20
        )

        # Historical trend line (model fit)
        plt.plot(
            fruit_df["Year"],
            y_pred_all,
            label="Predicted Trend (Historical)",
            linewidth=2
        )

        # Future predicted yield
        plt.scatter(
            future_years,
            future_yield_pred,
            label="Forecast (2026-2030)",
            color="green"
        )

        plt.title(f"{country_name} - {fruit}")
        plt.xlabel("Year")
        plt.ylabel("Yield")
        plt.legend()
        plt.grid(True)

        # Save plot
        plt.savefig(os.path.join(country_folder, f"{fruit}.png"))
        plt.close()

    print(f"✅ Done: {country_name} → saved in {country_folder}")





