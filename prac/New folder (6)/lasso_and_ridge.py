import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════
# FILE SETTINGS
# ═══════════════════════════════════════════════
FILE_PATH = "predict.xlsx"

# Your dataset has only ONE feature column: x
FEATURE_COLUMNS = ['x']

# Target column
TARGET_COLUMN = 'y'

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
if FILE_PATH.endswith(".csv"):
    df = pd.read_csv(FILE_PATH)
else:
    df = pd.read_excel(FILE_PATH)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("✅ Dataset Loaded!")

print("\nDataset:")
print(df)

# ─────────────────────────────────────────
# 2. REMOVE MISSING VALUES
# ─────────────────────────────────────────
df = df.dropna()

# ─────────────────────────────────────────
# 3. FEATURES & TARGET
# ─────────────────────────────────────────
X = df[FEATURE_COLUMNS].values

y = df[TARGET_COLUMN].values

# ─────────────────────────────────────────
# 4. FEATURE SCALING
# ─────────────────────────────────────────
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────
# 5. TRAIN TEST SPLIT
# ─────────────────────────────────────────
n_samples = len(df)

if n_samples <= 5:
    test_size = 0.4
else:
    test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=test_size,
    random_state=42
)

print("\n📊 Train Size:", len(X_train))
print("📊 Test Size :", len(X_test))

# ─────────────────────────────────────────
# 6. MODELS
# ─────────────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

results = []

# ─────────────────────────────────────────
# 7. TRAIN & EVALUATE
# ─────────────────────────────────────────
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n📘 {name}")

    print("MSE  :", round(mse, 4))

    print("RMSE :", round(rmse, 4))

    print("MAE  :", round(mae, 4))

    if len(y_test) >= 2:

        r2 = r2_score(y_test, y_pred)

        print("R²   :", round(r2, 4))

    else:

        r2 = np.nan

        print("R²   : Not enough test samples")

    results.append({
        'Model': name,
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R²': r2
    })

# ─────────────────────────────────────────
# 8. COMPARISON TABLE
# ─────────────────────────────────────────
results_df = pd.DataFrame(results)

print("\n📊 Model Comparison")

print(results_df)

# ─────────────────────────────────────────
# 9. COEFFICIENT COMPARISON
# ─────────────────────────────────────────
coef_df = pd.DataFrame({
    'Feature': FEATURE_COLUMNS,
    'Linear Regression':
        LinearRegression().fit(X_train, y_train).coef_,

    'Ridge Regression':
        Ridge(alpha=1.0).fit(X_train, y_train).coef_,

    'Lasso Regression':
        Lasso(alpha=0.1).fit(X_train, y_train).coef_
})

print("\n📘 Coefficient Comparison")

print(coef_df)

# ─────────────────────────────────────────
# 10. BAR PLOT
# ─────────────────────────────────────────
coef_melted = coef_df.melt(
    id_vars='Feature',
    var_name='Model',
    value_name='Coefficient'
)

plt.figure(figsize=(7, 4))

sns.barplot(
    data=coef_melted,
    x='Feature',
    y='Coefficient',
    hue='Model'
)

plt.title("Coefficient Comparison")

plt.tight_layout()

plt.savefig("coefficient_comparison.png")

plt.show()

# ─────────────────────────────────────────
# 11. ACTUAL VS PREDICTED
# ─────────────────────────────────────────
best_model = Ridge(alpha=1.0)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

plt.figure(figsize=(7, 4))

plt.scatter(
    y_test,
    y_pred,
    color='blue',
    s=50
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red'
)

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title("Actual vs Predicted")

plt.tight_layout()

plt.savefig("actual_vs_predicted.png")

plt.show()

# ─────────────────────────────────────────
# 12. PREDICT NEW VALUE
# ─────────────────────────────────────────
print("\n🔮 Predict New Value")

x = float(input("Enter x: "))

new_data = np.array([[x]])

# Scale input
new_data_scaled = scaler.transform(new_data)

prediction = best_model.predict(new_data_scaled)

print(
    "\n✅ Predicted y =",
    round(prediction[0], 4)
)

print("\n✅ Program Completed Successfully!")