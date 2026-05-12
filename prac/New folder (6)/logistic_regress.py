import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ═══════════════════════════════════════════════
# FILE SETTINGS
# ═══════════════════════════════════════════════
FILE_PATH = "predict.xlsx"

FEATURE_COLUMNS = ['x1', 'x2', 'x3']

TARGET_COLUMN = 'target'

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
if FILE_PATH.endswith(".csv"):
    df = pd.read_csv(FILE_PATH)
else:
    df = pd.read_excel(FILE_PATH)

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

y = df[TARGET_COLUMN]

# ─────────────────────────────────────────
# 4. ENCODE TARGET
# ─────────────────────────────────────────
le = LabelEncoder()

y_encoded = le.fit_transform(y)

classes = le.classes_

print("\n🎯 Classes:", classes)

# ─────────────────────────────────────────
# 5. FEATURE SCALING
# ─────────────────────────────────────────
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────
# 6. TRAIN TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42
)

print("\n📊 Train Size:", len(X_train))
print("📊 Test Size :", len(X_test))

# ─────────────────────────────────────────
# 7. TRAIN MODEL
# ─────────────────────────────────────────
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ─────────────────────────────────────────
# 8. PREDICTIONS
# ─────────────────────────────────────────
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Accuracy:", round(accuracy, 4))

# ─────────────────────────────────────────
# 9. CLASSIFICATION REPORT
# ─────────────────────────────────────────
print("\n📘 Classification Report")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.unique(y_test),
        zero_division=0
    )
)

# ─────────────────────────────────────────
# 10. CONFUSION MATRIX
# ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

print("\n📘 Confusion Matrix")

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap='Blues')

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

# ─────────────────────────────────────────
# 11. FEATURE IMPORTANCE
# ─────────────────────────────────────────
coef_df = pd.DataFrame({
    'Feature': FEATURE_COLUMNS,
    'Coefficient': model.coef_[0]
})

print("\n📘 Feature Importance")

print(coef_df)

plt.figure(figsize=(7, 4))

sns.barplot(
    data=coef_df,
    x='Feature',
    y='Coefficient'
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()

# ─────────────────────────────────────────
# 12. CROSS VALIDATION
# ─────────────────────────────────────────
if len(df) >= 10:

    cv = min(10, len(df))

    cv_scores = cross_val_score(
        LogisticRegression(max_iter=1000),
        X_scaled,
        y_encoded,
        cv=cv,
        scoring='accuracy'
    )

    print(f"\n🔁 {cv}-Fold Cross Validation")

    print("Scores :", cv_scores.round(3))

    print("Mean Accuracy :", round(cv_scores.mean(), 4))

# ─────────────────────────────────────────
# 13. PREDICT NEW SAMPLE
# ─────────────────────────────────────────
print("\n🔮 Predict New Sample")

new_values = []

for col in FEATURE_COLUMNS:

    val = float(input(f"Enter {col}: "))

    new_values.append(val)

new_data = np.array([new_values])

new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)

prediction_prob = model.predict_proba(
    new_data_scaled
)[0]

predicted_class = le.inverse_transform(
    prediction
)[0]

print("\n✅ Predicted Class:", predicted_class)

print("\n📊 Probabilities")

for cls, prob in zip(classes, prediction_prob):

    print(f"{cls} : {round(prob, 4)}")

print("\n✅ Logistic Regression Completed!")