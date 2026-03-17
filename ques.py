"""
ANN FROM SCRATCH - Fruit Yield Prediction
Network: Input(2) -> Hidden(8, ReLU) -> Output(1, Linear)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. LOAD & PREPROCESS DATA
# -----------------------------------------
df = pd.read_csv("fruit_yield_raw.csv")

area_df  = df[df["Element"] == "Area harvested"][["Area","Item","Year","Value"]].rename(columns={"Value":"Area_ha"})
yield_df = df[df["Element"] == "Yield"         ][["Area","Item","Year","Value"]].rename(columns={"Value":"Yield_kgha"})
data = pd.merge(area_df, yield_df, on=["Area","Item","Year"]).dropna()

X_raw = data[["Year", "Area_ha"]].values.astype(float)
y_raw = data["Yield_kgha"].values.astype(float)

# -----------------------------------------
# 2. NORMALISE (Min-Max -> [0, 1])
# -----------------------------------------
X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
y_min, y_max = y_raw.min(), y_raw.max()

X = (X_raw - X_min) / (X_max - X_min)
y = ((y_raw - y_min) / (y_max - y_min)).reshape(-1, 1)

# Train / Test split (80 / 20)
np.random.seed(42)
idx   = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

# -----------------------------------------
# 3. INITIALISE WEIGHTS (He initialisation)
# -----------------------------------------
input_size, hidden_size, output_size = 2, 8, 1

np.random.seed(0)
W1 = np.random.randn(input_size,  hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, output_size))

# -----------------------------------------
# 4. ACTIVATION & LOSS FUNCTIONS
# -----------------------------------------
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# -----------------------------------------
# 5. TRAINING LOOP (Forward + Backward)
# -----------------------------------------
learning_rate = 0.01
epochs        = 2000
train_losses, test_losses = [], []

print(f"Training for {epochs} epochs...\n")

for epoch in range(1, epochs + 1):
    N = X_train.shape[0]

    # -- FORWARD PASS --
    z1         = X_train @ W1 + b1    # (N, 8)
    a1         = relu(z1)             # (N, 8)
    y_pred     = a1 @ W2 + b2        # (N, 1)
    loss_train = mse_loss(y_pred, y_train)

    # -- BACKWARD PASS --
    # Output layer gradients
    dL_dz2 = (2 / N) * (y_pred - y_train)          # (N, 1)
    dL_dW2 = a1.T @ dL_dz2                          # (8, 1)
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)  # (1, 1)

    # Hidden layer gradients
    dL_da1 = dL_dz2 @ W2.T                          # (N, 8)
    dL_dz1 = dL_da1 * relu_derivative(z1)           # (N, 8)
    dL_dW1 = X_train.T @ dL_dz1                     # (2, 8)
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)  # (1, 8)

    # -- WEIGHT UPDATE --
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1

    # Test loss
    y_pred_test = relu(X_test @ W1 + b1) @ W2 + b2
    loss_test   = mse_loss(y_pred_test, y_test)

    train_losses.append(loss_train)
    test_losses.append(loss_test)

    if epoch % 200 == 0 or epoch == 1:
        print(f"  Epoch {epoch:4d}  |  Train MSE: {loss_train:.6f}  |  Test MSE: {loss_test:.6f}")

# -----------------------------------------
# 6. EVALUATE
# -----------------------------------------
y_pred_norm   = relu(X_test @ W1 + b1) @ W2 + b2
y_pred_actual = y_pred_norm * (y_max - y_min) + y_min
y_test_actual = y_test      * (y_max - y_min) + y_min

mae    = np.mean(np.abs(y_pred_actual - y_test_actual))
rmse   = np.sqrt(np.mean((y_pred_actual - y_test_actual) ** 2))
r2     = 1 - np.sum((y_test_actual - y_pred_actual) ** 2) / np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)

print(f"\nMAE  : {mae:.2f} kg/ha")
print(f"RMSE : {rmse:.2f} kg/ha")
print(f"R2   : {r2:.4f}")

# -----------------------------------------
# 7. PLOTS
# -----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ANN - Fruit Yield Prediction", fontsize=14, fontweight="bold")

axes[0].plot(train_losses, label="Train MSE", color="steelblue")
axes[0].plot(test_losses,  label="Test MSE",  color="orangered", linestyle="--")
axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

min_v, max_v = float(y_test_actual.min()), float(y_test_actual.max())
axes[1].scatter(y_test_actual, y_pred_actual, alpha=0.4, color="steelblue", s=20)
axes[1].plot([min_v, max_v], [min_v, max_v], "r--", label="Perfect fit")
axes[1].set_title(f"Predicted vs Actual  (R2 = {r2:.4f})")
axes[1].set_xlabel("Actual Yield (kg/ha)")
axes[1].set_ylabel("Predicted Yield (kg/ha)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ann_fruit_yield_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nPlot saved -> ann_fruit_yield_results.png")