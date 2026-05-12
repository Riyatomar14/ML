import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    label_binarize
)

# ═══════════════════════════════════════════════════════
# FILE DETAILS
# ═══════════════════════════════════════════════════════
FILE_PATH = "naive_bayes_student_dataset.xlsx"
TARGET_COLUMN = "Pass/Fail"

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
if FILE_PATH.endswith(".csv"):
    df = pd.read_csv(FILE_PATH)
else:
    df = pd.read_excel(FILE_PATH)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("✅ Dataset loaded!")
print("Shape :", df.shape)
print("Columns :", list(df.columns))
print(df.head())

# ─────────────────────────────────────────
# 2. REMOVE MISSING VALUES
# ─────────────────────────────────────────
df = df.dropna()

# ─────────────────────────────────────────
# 3. SPLIT FEATURES & TARGET
# ─────────────────────────────────────────
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# ─────────────────────────────────────────
# 4. ENCODE TARGET
# ─────────────────────────────────────────
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

classes = le_target.classes_
n_classes = len(classes)

print("\n🎯 Classes :", classes)

# ─────────────────────────────────────────
# 5. DETECT COLUMN TYPES
# ─────────────────────────────────────────
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\n🔢 Numeric Columns :", numeric_cols)
print("🔤 Categorical Columns :", categorical_cols)

# ─────────────────────────────────────────
# 6. ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────
X_processed = X.copy()

oe = OrdinalEncoder()
X_processed[categorical_cols] = oe.fit_transform(X[categorical_cols])

X_processed = X_processed.astype(float)

# ─────────────────────────────────────────
# 7. SELECT MODEL
# ─────────────────────────────────────────
if categorical_cols and not numeric_cols:
    model = CategoricalNB()
    model_name = "CategoricalNB"
else:
    model = GaussianNB()
    model_name = "GaussianNB"

print(f"\n🤖 Model Selected : {model_name}")

# ─────────────────────────────────────────
# 8. TRAIN TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y_encoded,
    test_size=0.33,
    random_state=42
)

print(f"\n📊 Train Size : {len(X_train)}")
print(f"📊 Test Size  : {len(X_test)}")

# ─────────────────────────────────────────
# 9. TRAIN MODEL
# ─────────────────────────────────────────
model.fit(X_train, y_train)

# ─────────────────────────────────────────
# 10. PREDICTIONS
# ─────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# ─────────────────────────────────────────
# 11. ACCURACY
# ─────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy : {round(acc, 4)}")

# ─────────────────────────────────────────
# 12. CLASSIFICATION REPORT
# ─────────────────────────────────────────
print("\n📋 Classification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=classes,
        zero_division=0
    )
)

# ─────────────────────────────────────────
# 13. CONFUSION MATRIX
# ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {model_name}")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

print("💾 Confusion matrix saved!")

# ─────────────────────────────────────────
# 14. AUC ROC SCORE
# ─────────────────────────────────────────
try:
    if n_classes == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
        print(f"\n📈 AUC ROC Score : {round(auc, 4)}")

except Exception as e:
    print("\n⚠️ AUC skipped :", e)

# ─────────────────────────────────────────
# 15. CROSS VALIDATION
# ─────────────────────────────────────────
cv_scores = cross_val_score(
    model,
    X_processed,
    y_encoded,
    cv=3,
    scoring='accuracy'
)

print("\n🔁 Cross Validation Scores :")
print(cv_scores)

print("\nMean Accuracy :", round(cv_scores.mean(), 4))

# ─────────────────────────────────────────
# 16. NEW SAMPLE PREDICTION
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("🔮 PREDICT NEW SAMPLE")
print("=" * 50)

new_sample = {}

for col in X.columns:

    val = input(f"{col}: ").strip()

    new_sample[col] = val

new_df = pd.DataFrame([new_sample])

# Encode categorical values
new_df[categorical_cols] = oe.transform(
    new_df[categorical_cols]
)

new_df = new_df.astype(float)

prediction = model.predict(new_df)

predicted_class = le_target.inverse_transform(prediction)[0]

probabilities = model.predict_proba(new_df)[0]

print(f"\n✅ Predicted Class : {predicted_class}")

print("\n📊 Probabilities :")

for cls, prob in zip(classes, probabilities):
    print(f"{cls} : {round(prob, 4)}")