import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import (
    StandardScaler,
    label_binarize
)

# ─────────────────────────────────────────
# 1. LOAD EXCEL FILE
# ─────────────────────────────────────────

file_name = "data.xlsx"

df = pd.read_excel(file_name)

print("\n📘 Dataset")
print(df.head())

print("\nShape:", df.shape)

# ─────────────────────────────────────────
# 2. FEATURES & TARGET
# ─────────────────────────────────────────

target_column = "target"

X = df.drop(target_column, axis=1).values
y = df[target_column].values

print("\nTarget Classes:")
print(np.unique(y))

# ─────────────────────────────────────────
# 3. FEATURE SCALING
# ─────────────────────────────────────────

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n✅ Feature Scaling Done")

# ─────────────────────────────────────────
# 4. TRAIN TEST SPLIT
# ─────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

# ─────────────────────────────────────────
# 5. FIND BEST K
# ─────────────────────────────────────────

k_values = range(1, 21)

train_acc = []
test_acc = []

for k in k_values:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(X_train, y_train)

    train_acc.append(
        accuracy_score(
            y_train,
            knn.predict(X_train)
        )
    )

    test_acc.append(
        accuracy_score(
            y_test,
            knn.predict(X_test)
        )
    )

# ─────────────────────────────────────────
# 6. PLOT ACCURACY VS K
# ─────────────────────────────────────────

plt.figure(figsize=(8,5))

plt.plot(
    k_values,
    train_acc,
    marker='o',
    label='Train Accuracy'
)

plt.plot(
    k_values,
    test_acc,
    marker='s',
    label='Test Accuracy'
)

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.xticks(k_values)
plt.legend()

plt.show()

# ─────────────────────────────────────────
# 7. BEST K
# ─────────────────────────────────────────

best_k = k_values[
    np.argmax(test_acc)
]

print("\n✅ Best K:", best_k)

# ─────────────────────────────────────────
# 8. TRAIN FINAL MODEL
# ─────────────────────────────────────────

model = KNeighborsClassifier(
    n_neighbors=best_k
)

model.fit(
    X_train,
    y_train
)

print("\n✅ KNN Model Trained")

# ─────────────────────────────────────────
# 9. PREDICTION
# ─────────────────────────────────────────

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)

# ─────────────────────────────────────────
# 10. ACCURACY
# ─────────────────────────────────────────

acc = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:",
      round(acc, 4))

# ─────────────────────────────────────────
# 11. CLASSIFICATION REPORT
# ─────────────────────────────────────────

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ─────────────────────────────────────────
# 12. CONFUSION MATRIX
# ─────────────────────────────────────────

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    f"KNN Confusion Matrix (K={best_k})"
)

plt.show()

# ─────────────────────────────────────────
# 13. MANUAL METRICS
# ─────────────────────────────────────────

TP = np.diag(cm)

FP = cm.sum(axis=0) - TP

FN = cm.sum(axis=1) - TP

TN = cm.sum() - (
    TP + FP + FN
)

accuracy = (
    TP + TN
) / (
    TP + TN + FP + FN
)

precision = TP / (
    TP + FP
)

recall = TP / (
    TP + FN
)

specificity = TN / (
    TN + FP
)

f1 = (
    2 * precision * recall
) / (
    precision + recall
)

print("\n📘 Per Class Metrics")

for i in range(len(TP)):

    print(f"\nClass {i}")

    print("TP :", TP[i])
    print("TN :", TN[i])
    print("FP :", FP[i])
    print("FN :", FN[i])

    print(
        "Precision:",
        round(precision[i], 4)
    )

    print(
        "Recall:",
        round(recall[i], 4)
    )

    print(
        "Specificity:",
        round(specificity[i], 4)
    )

    print(
        "F1 Score:",
        round(f1[i], 4)
    )

# ─────────────────────────────────────────
# 14. ROC AUC
# ─────────────────────────────────────────

classes = np.unique(y)

y_bin = label_binarize(
    y_test,
    classes=classes
)

auc = roc_auc_score(
    y_bin,
    y_prob,
    multi_class='ovr',
    average='macro'
)

print(
    "\nAUC ROC:",
    round(auc, 4)
)

# ─────────────────────────────────────────
# 15. ROC CURVE
# ─────────────────────────────────────────

plt.figure(figsize=(7,5))

for i in range(len(classes)):

    fpr, tpr, _ = roc_curve(
        y_bin[:, i],
        y_prob[:, i]
    )

    plt.plot(
        fpr,
        tpr,
        label=f'Class {classes[i]}'
    )

plt.plot(
    [0,1],
    [0,1],
    '--'
)

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()

plt.show()

# ─────────────────────────────────────────
# 16. DISTANCE METRIC
# ─────────────────────────────────────────

metrics = [
    'euclidean',
    'manhattan',
    'minkowski'
]

print("\nDistance Metric Comparison")

for metric in metrics:

    knn = KNeighborsClassifier(
        n_neighbors=best_k,
        metric=metric
    )

    knn.fit(
        X_train,
        y_train
    )

    acc = accuracy_score(
        y_test,
        knn.predict(X_test)
    )

    print(
        metric,
        "Accuracy:",
        round(acc, 4)
    )

# ─────────────────────────────────────────
# 17. 10-FOLD CROSS VALIDATION
# ─────────────────────────────────────────

cv_acc = cross_val_score(
    KNeighborsClassifier(
        n_neighbors=best_k
    ),
    X_scaled,
    y,
    cv=10,
    scoring='accuracy'
)

print(
    "\n10 Fold Accuracy:",
    cv_acc.round(3)
)

print(
    "Mean Accuracy:",
    round(cv_acc.mean(), 4)
)