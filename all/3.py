import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.describe())

# 2. Check missing values
print("\nMissing values:", df.isnull().sum().sum())

# 3. Correlation Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# 4. Separate features and target
X = df.drop('Price', axis=1).values
y = df['Price'].values

# 5. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 7. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Print coefficients
print("\n--- Model Coefficients ---")
for name, coef in zip(data.feature_names, model.coef_):
    print(f"  {name:15s}: {round(coef, 4)}")
print(f"  Intercept      : {round(model.intercept_, 4)}")

# 9. Predict
y_pred = model.predict(X_test)

# 10. Evaluation Metrics
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

# Adjusted R²
n = X_test.shape[0]
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\n--- Evaluation Metrics ---")
print("MSE         :", round(mse, 4))
print("RMSE        :", round(rmse, 4))
print("MAE         :", round(mae, 4))
print("R²          :", round(r2, 4))
print("Adjusted R² :", round(adj_r2, 4))

# 11. Actual vs Predicted Plot
plt.figure(figsize=(7, 4))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue', s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2, label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted - Multiple Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# 12. Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(7, 4))
plt.scatter(y_pred, residuals, alpha=0.3, color='purple', s=10)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("residual_plot.png")
plt.show()

# 13. Feature Importance (by coefficient magnitude)
coef_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

plt.figure(figsize=(7, 4))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm')
plt.title("Feature Importance by Coefficient")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("\nFeature Importance:")
print(coef_df.to_string(index=False))

# 14. Cross Validation (10-fold)
cv_r2  = cross_val_score(LinearRegression(), X_scaled, y, cv=10, scoring='r2')
cv_mse = cross_val_score(LinearRegression(), X_scaled, y,
                          cv=10, scoring='neg_mean_squared_error')

print("\n10-Fold CV R² Scores:", cv_r2.round(3))
print("Mean R²  :", round(cv_r2.mean(), 4))
print("Mean MSE :", round(-cv_mse.mean(), 4))