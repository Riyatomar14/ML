import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
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

FEATURE_COLUMNS = ['x1', 'x2', 'x3']

TARGET_COLUMN = 'y'

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

print("\nShape:", df.shape)

print("\nDataset:")
print(df)

# ─────────────────────────────────────────
# 2. CHECK MISSING VALUES
# ─────────────────────────────────────────
print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()

# ─────────────────────────────────────────
# 3. CORRELATION HEATMAP
# ─────────────────────────────────────────
plt.figure(figsize=(7, 5))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.show()

# ─────────────────────────────────────────
# 4. FEATURES & TARGET
# ─────────────────────────────────────────
X = df[FEATURE_COLUMNS].values

y = df[TARGET_COLUMN].values

# ─────────────────────────────────────────
# 5. FEATURE SCALING
# ─────────────────────────────────────────
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────
# 6. TRAIN TEST SPLIT
# Works for small datasets also
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
# 7. TRAIN MODEL
# ─────────────────────────────────────────
model = LinearRegression()

model.fit(X_train, y_train)

# ─────────────────────────────────────────
# 8. PRINT COEFFICIENTS
# ─────────────────────────────────────────
print("\n📘 Model Coefficients")

for feature, coef in zip(FEATURE_COLUMNS, model.coef_):

    print(f"{feature} : {round(coef, 4)}")

print("Intercept :", round(model.intercept_, 4))

# Equation
print("\nEquation:")

equation = f"y = {round(model.intercept_,2)}"

for feature, coef in zip(FEATURE_COLUMNS, model.coef_):

    equation += f" + ({round(coef,2)} * {feature})"

print(equation)

# ─────────────────────────────────────────
# 9. PREDICTION
# ─────────────────────────────────────────
y_pred = model.predict(X_test)

# ─────────────────────────────────────────
# 10. EVALUATION METRICS
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
# 11. ACTUAL VS PREDICTED
# ─────────────────────────────────────────
plt.figure(figsize=(7, 4))

plt.scatter(
    y_test,
    y_pred,
    color='blue',
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
    color='purple',
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
# 13. FEATURE IMPORTANCE
# ─────────────────────────────────────────
coef_df = pd.DataFrame({
    'Feature': FEATURE_COLUMNS,
    'Coefficient': model.coef_
})

coef_df = coef_df.sort_values(
    by='Coefficient',
    key=abs,
    ascending=False
)

print("\n📊 Feature Importance")

print(coef_df)

# ─────────────────────────────────────────
# 14. PREDICT NEW DATA
# ─────────────────────────────────────────
print("\n🔮 Predict New Value")

new_values = []

for feature in FEATURE_COLUMNS:

    val = float(input(f"Enter {feature}: "))

    new_values.append(val)

new_data = np.array([new_values])

# Scale new data
new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)

print("\n✅ Predicted y =", round(prediction[0], 4))