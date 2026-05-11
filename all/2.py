import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Load dataset (built-in - no download needed)
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

# 2. Use only ONE feature for Simple Linear Regression
# Using 'AveRooms' (average number of rooms) to predict house price
X = df[['AveRooms']].values
y = df['Price'].values

# 3. Visualize the data
plt.figure(figsize=(7, 4))
plt.scatter(X, y, alpha=0.3, color='steelblue', s=10)
plt.xlabel("Average Rooms")
plt.ylabel("House Price")
plt.title("Scatter Plot - Rooms vs Price")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()

# 4. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Print coefficients
print("Intercept (b0):", round(model.intercept_, 4))
print("Slope (b1):", round(model.coef_[0], 4))
print("Equation: Price =", round(model.intercept_, 2),
      "+", round(model.coef_[0], 2), "* AveRooms")

# 7. Predict
y_pred = model.predict(X_test)

# 8. Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluation Metrics ---")
print("MSE  :", round(mse, 4))
print("RMSE :", round(rmse, 4))
print("MAE  :", round(mae, 4))
print("R²   :", round(r2, 4))

# 9. Plot Regression Line
plt.figure(figsize=(7, 4))
plt.scatter(X_test, y_test, alpha=0.3, color='steelblue', s=10, label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Average Rooms")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("regression_line.png")
plt.show()

# 10. Residual Plot
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

# 11. Cross Validation (10-fold)
cv_scores = cross_val_score(LinearRegression(), X, y, cv=10, scoring='r2')
print("\n10-Fold CV R² Scores:", cv_scores.round(3))
print("Mean R²:", round(cv_scores.mean(), 4))
print("Std Dev:", round(cv_scores.std(), 4))