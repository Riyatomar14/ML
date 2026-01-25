import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ================================
# LOAD CLEANED DATA
# ================================
df = pd.read_csv("fruit_yield_cleaned.csv")

# ================================
# COUNTRIES TO PROCESS
# ================================
countries = {
    "Australia": "Australia",
    "India": "India",
    "China": "China",
    "USA": "United States of America",
    "South Africa": "South Africa"
}

# ================================
# FUNCTION: MODEL + PREDICT + METRICS
# ================================
def analyze_fruit(fruit_df):
    X = fruit_df["Year"].values.reshape(-1, 1)
    Y = fruit_df["Value"].values

    # -------------------------
    # TRAIN-TEST SPLIT
    # -------------------------
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, Y_train)  # training

    m = model.coef_[0]
    c = model.intercept_

    # -------------------------
    # EVALUATION (ON TEST DATA)
    # -------------------------
    Y_test_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_test_pred)
    r2 = r2_score(Y_test, Y_test_pred)

    # Predict 2026-2030
    future_years = [[year] for year in range(2026, 2031)]
    future_pred = model.predict(future_years)

    return m, c, mse, r2, future_pred, model

# ================================
# PROCESS EACH COUNTRY
# ================================
for country_name, country_value in countries.items():
    country_df = df[df["Area"] == country_value]

    if country_df.empty:
        print(f"⚠️ {country_name} not found in dataset")
        continue

    # Create folder for country
    folder_name = country_name.replace(" ", "_")
    os.makedirs(folder_name, exist_ok=True)

    fruits = country_df["Item"].unique()
    results = []

    for fruit in fruits:
        fruit_df = country_df[country_df["Item"] == fruit]

        if fruit_df.shape[0] < 5:
            continue

        m, c, mse, r2, future_pred, model = analyze_fruit(fruit_df)

        # Save results in list
        results.append({
            "Fruit": fruit,
            "m": m,
            "c": c,
            "MSE": mse,
            "R2": r2,
            "2026": future_pred[0],
            "2027": future_pred[1],
            "2028": future_pred[2],
            "2029": future_pred[3],
            "2030": future_pred[4]
        })

        # ================================
        # SAVE GRAPH FOR EACH FRUIT (2026-2030)
        # ================================
        plt.figure(figsize=(8, 5))

        # Actual data points
        plt.scatter(fruit_df["Year"], fruit_df["Value"], label="Actual Data")

        # Line from historical + future years
        all_years = list(fruit_df["Year"]) + list(range(2026, 2031))
        all_years_sorted = sorted(all_years)

        # Predict for all years
        X_all = pd.DataFrame({"Year": all_years_sorted})["Year"].values.reshape(-1, 1)
        y_all_pred = model.predict(X_all)

        plt.plot(all_years_sorted, y_all_pred, label="Regression Line (2026-2030)", color="red")

        # Mark predicted points
        for year, pred in zip(range(2026, 2031), future_pred):
            plt.scatter(year, pred, color="green")

        plt.title(f"{country_name} - {fruit}")
        plt.xlabel("Year")
        plt.ylabel("Yield Value")
        plt.legend()
        plt.grid(True)

        # Save PNG inside country folder
        plt.savefig(f"{folder_name}/{fruit}.png")
        plt.close()

    # Save CSV for country
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{folder_name}/{country_name}_results.csv", index=False)

    print(f"✅ Saved {country_name} folder with CSV + PNGs")



