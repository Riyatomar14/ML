import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ═══════════════════════════════════════════════
# FILE SETTINGS
# ═══════════════════════════════════════════════
FILE_PATH = "predict.xlsx"

FEATURE_COLUMN = "x"
TARGET_COLUMN = "y"

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
if FILE_PATH.endswith(".csv"):
    df = pd.read_csv(FILE_PATH)
else:
    df = pd.read_excel(FILE_PATH)

# Remove extra spaces
df.columns = df.columns.str.strip()

print("✅ Dataset Loaded!")

print("Shape :", df.shape)

print("Columns :", list(df.columns))

print("\nDataset:")
print(df)

# ─────────────────────────────────────────
# 2. REMOVE MISSING VALUES
# ─────────────────────────────────────────
df = df.dropna()

# ─────────────────────────────────────────
# 3. SELECT FEATURE & TARGET
# ─────────────────────────────────────────
X = df[[FEATURE_COLUMN]].values
y = df[TARGET_COLUMN].values

# ─────────────────────────────────────────
# 4. SCATTER PLOT
# ─────────────────────────────────────────
plt.figure(figsize=(7, 4))

plt.scatter(
    X,
    y,
    color='blue',
    s=40
)

plt.xlabel(FEATURE_COLUMN)
plt.ylabel(TARGET_COLUMN)

plt.title("Scatter Plot")

plt.tight_layout()

plt.savefig("scatter_plot.png")

plt.show()

# ─────────────────────────────────────────
# 5. TRAIN TEST SPLIT
# ─────────────────────────────────────────
n_samples = len(df)

if n_samples <= 5:
    test_size = 0.4
else:
    test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

print("\n📊 Train Size :", len(X_train))
print("📊 Test Size  :", len(X_test))

# ─────────────────────────────────────────
# 6. TRAIN MODEL
# ─────────────────────────────────────────
model = LinearRegression()

model.fit(X_train, y_train)

# ─────────────────────────────────────────
# 7. PRINT EQUATION
# ─────────────────────────────────────────
print("\n📘 Linear Regression Equation")

print("Intercept (b0):", round(model.intercept_, 4))

print("Slope (b1):", round(model.coef_[0], 4))

print(
    f"Equation: {TARGET_COLUMN} =",
    round(model.intercept_, 2),
    "+",
    round(model.coef_[0], 2),
    "*",
    FEATURE_COLUMN
)

# ─────────────────────────────────────────
# 8. PREDICTION ON TEST DATA
# ─────────────────────────────────────────
y_pred = model.predict(X_test)

# ─────────────────────────────────────────
# 9. EVALUATION METRICS
# ─────────────────────────────────────────
mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test, y_pred)

print("\n📊 Evaluation Metrics")

print("MSE  :", round(mse, 4))

print("RMSE :", round(rmse, 4))

print("MAE  :", round(mae, 4))

# R² only if test set has 2+ samples
if len(y_test) >= 2:

    r2 = r2_score(y_test, y_pred)

    print("R²   :", round(r2, 4))

else:

    print("R²   : Not enough test samples")

# ─────────────────────────────────────────
# 10. REGRESSION LINE
# ─────────────────────────────────────────
plt.figure(figsize=(7, 4))

plt.scatter(
    X_test,
    y_test,
    color='blue',
    s=40,
    label='Actual'
)

sorted_idx = X_test[:, 0].argsort()

plt.plot(
    X_test[sorted_idx],
    y_pred[sorted_idx],
    color='red',
    linewidth=2,
    label='Regression Line'
)

plt.xlabel(FEATURE_COLUMN)

plt.ylabel(TARGET_COLUMN)

plt.title("Simple Linear Regression")

plt.legend()

plt.tight_layout()

plt.savefig("regression_line.png")

plt.show()

# ─────────────────────────────────────────
# 11. RESIDUAL PLOT
# ─────────────────────────────────────────
residuals = y_test - y_pred

plt.figure(figsize=(7, 4))

plt.scatter(
    y_pred,
    residuals,
    color='purple',
    s=40
)

plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel("Predicted Values")

plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig("residual_plot.png")

plt.show()

# ─────────────────────────────────────────
# 12. PREDICT NEW VALUE
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("🔮 PREDICT NEW VALUE")
print("=" * 50)

new_value = float(
    input(f"Enter {FEATURE_COLUMN}: ")
)

new_data = np.array([[new_value]])

predicted_price = model.predict(new_data)

print(
    f"\n✅ Predicted {TARGET_COLUMN}:",
    round(predicted_price[0], 4)
)

print("\n✅ Program executed successfully!")