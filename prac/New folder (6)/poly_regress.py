import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

# ═══════════════════════════════════════════════
# FILE SETTINGS
# ═══════════════════════════════════════════════
FILE_PATH = "predict.xlsx"

FEATURE_COLUMN = "x"

TARGET_COLUMN = "y"

POLYNOMIAL_DEGREE = 2

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
if FILE_PATH.endswith(".csv"):
    df = pd.read_csv(FILE_PATH)
else:
    df = pd.read_excel(FILE_PATH)

# Remove spaces
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
X = df[[FEATURE_COLUMN]].values

y = df[TARGET_COLUMN].values

# ─────────────────────────────────────────
# 4. VISUALIZE DATA
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

plt.title("Polynomial Regression Data")

plt.tight_layout()

plt.savefig("raw_data.png")

plt.show()

# ─────────────────────────────────────────
# 5. TRAIN TEST SPLIT
# Works for small datasets
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

print("\n📊 Train Size:", len(X_train))
print("📊 Test Size :", len(X_test))

# ─────────────────────────────────────────
# 6. BUILD MODEL
# ─────────────────────────────────────────
model = Pipeline([
    ('poly', PolynomialFeatures(degree=POLYNOMIAL_DEGREE)),
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

# ─────────────────────────────────────────
# 7. TRAIN MODEL
# ─────────────────────────────────────────
model.fit(X_train, y_train)

# ─────────────────────────────────────────
# 8. PREDICTION
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

# R² only if enough samples
if len(y_test) >= 2:

    r2 = r2_score(y_test, y_pred)

    print("R²   :", round(r2, 4))

else:

    print("R²   : Not enough test samples")

# ─────────────────────────────────────────
# 10. POLYNOMIAL CURVE
# ─────────────────────────────────────────
X_plot = np.linspace(
    X.min(),
    X.max(),
    300
).reshape(-1, 1)

y_plot = model.predict(X_plot)

plt.figure(figsize=(7, 4))

plt.scatter(
    X,
    y,
    color='blue',
    s=40,
    label='Actual Data'
)

plt.plot(
    X_plot,
    y_plot,
    color='red',
    linewidth=2,
    label=f'Degree {POLYNOMIAL_DEGREE}'
)

plt.xlabel(FEATURE_COLUMN)

plt.ylabel(TARGET_COLUMN)

plt.title("Polynomial Regression Curve")

plt.legend()

plt.tight_layout()

plt.savefig("polynomial_curve.png")

plt.show()

# ─────────────────────────────────────────
# 11. ACTUAL VS PREDICTED
# ─────────────────────────────────────────
plt.figure(figsize=(7, 4))

plt.scatter(
    y_test,
    y_pred,
    color='purple',
    s=40
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
# 12. RESIDUAL PLOT
# ─────────────────────────────────────────
residuals = y_test - y_pred

plt.figure(figsize=(7, 4))

plt.scatter(
    y_pred,
    residuals,
    color='green',
    s=40
)

plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig("residual_plot.png")

plt.show()

# ─────────────────────────────────────────
# 13. PREDICT NEW VALUE
# ─────────────────────────────────────────
print("\n🔮 Predict New Value")

new_x = float(input(f"Enter {FEATURE_COLUMN}: "))

new_data = np.array([[new_x]])

prediction = model.predict(new_data)

print(
    f"\n✅ Predicted {TARGET_COLUMN} =",
    round(prediction[0], 4)
)

print("\n✅ Polynomial Regression Completed!")