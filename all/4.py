import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# 1. Create a non-linear dataset (or use real one)
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = 2*X**2 + 3*X + np.random.randn(200, 1)*2  # quadratic pattern
y = y.ravel()

# 2. Visualize raw data
plt.figure(figsize=(7, 4))
plt.scatter(X, y, alpha=0.5, color='steelblue', s=15)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Raw Data - Non-linear Pattern")
plt.tight_layout()
plt.savefig("raw_data.png")
plt.show()

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Compare Linear vs Polynomial degrees
degrees = [1, 2, 3, 4]
results = []

plt.figure(figsize=(12, 8))
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

for i, deg in enumerate(degrees):
    # Build pipeline
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=deg)),
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_plot = model.predict(X_plot)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    results.append({
        'Degree': deg,
        'MSE'   : round(mse, 4),
        'RMSE'  : round(rmse, 4),
        'MAE'   : round(mae, 4),
        'R²'    : round(r2, 4)
    })

    # Plot
    plt.subplot(2, 2, i+1)
    plt.scatter(X_test, y_test, alpha=0.4, color='steelblue',
                s=15, label='Actual')
    plt.plot(X_plot, y_plot, color='red',
             linewidth=2, label=f'Degree {deg}')
    plt.title(f"Degree {deg} | R² = {round(r2, 3)}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend(fontsize=8)

plt.suptitle("Polynomial Regression - Degree Comparison", fontsize=13)
plt.tight_layout()
plt.savefig("degree_comparison.png")
plt.show()

# 5. Print comparison table
results_df = pd.DataFrame(results)
print("\n--- Degree Comparison ---")
print(results_df.to_string(index=False))

# 6. Best model (degree 2 for this data)
best_degree = 2
best_model = Pipeline([
    ('poly',   PolynomialFeatures(degree=best_degree)),
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

# 7. Final Evaluation
mse  = mean_squared_error(y_test, y_pred_best)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred_best)
r2   = r2_score(y_test, y_pred_best)

print(f"\n--- Best Model (Degree {best_degree}) ---")
print("MSE  :", round(mse, 4))
print("RMSE :", round(rmse, 4))
print("MAE  :", round(mae, 4))
print("R²   :", round(r2, 4))

# 8. Actual vs Predicted
plt.figure(figsize=(7, 4))
plt.scatter(y_test, y_pred_best, alpha=0.5,
            color='steelblue', s=15)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2, label='Perfect Prediction')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted - Degree {best_degree}")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# 9. Residual Plot
residuals = y_test - y_pred_best
plt.figure(figsize=(7, 4))
plt.scatter(y_pred_best, residuals,
            alpha=0.5, color='purple', s=15)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("residual_plot.png")
plt.show()

# 10. Overfitting check - Training vs Test R²
print("\n--- Overfitting Check ---")
for deg in range(1, 8):
    m = Pipeline([
        ('poly',   PolynomialFeatures(degree=deg)),
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ])
    m.fit(X_train, y_train)
    train_r2 = r2_score(y_train, m.predict(X_train))
    test_r2  = r2_score(y_test,  m.predict(X_test))
    print(f"Degree {deg} | Train R²: {round(train_r2,4)}"
          f" | Test R²: {round(test_r2,4)}"
          f" | {'OVERFIT' if train_r2 - test_r2 > 0.1 else 'OK'}")

# 11. Cross Validation
cv_scores = cross_val_score(
    Pipeline([
        ('poly',   PolynomialFeatures(degree=best_degree)),
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ]),
    X, y, cv=10, scoring='r2')

print(f"\n10-Fold CV R² (Degree {best_degree}):", cv_scores.round(3))
print("Mean R²:", round(cv_scores.mean(), 4))
print("Std Dev:", round(cv_scores.std(), 4))