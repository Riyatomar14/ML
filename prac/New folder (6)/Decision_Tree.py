import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
    export_text
)
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

X = df.drop(columns=[target_column])
y = df[target_column]

print("\nTarget Classes:")
print(y.unique())

# ─────────────────────────────────────────
# 3. TRAIN TEST SPLIT
# ─────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

# ─────────────────────────────────────────
# 4. TRAIN DECISION TREE
# ─────────────────────────────────────────

model = DecisionTreeClassifier(
    criterion='gini',
    random_state=42
)

model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully")

print("\nTree Depth:",
      model.get_depth())

print("Number of Leaves:",
      model.get_n_leaves())

# ─────────────────────────────────────────
# 5. PREDICTION
# ─────────────────────────────────────────

y_pred = model.predict(X_test)

# Probability
if len(np.unique(y)) == 2:
    y_prob = model.predict_proba(X_test)[:, 1]

# ─────────────────────────────────────────
# 6. ACCURACY
# ─────────────────────────────────────────

acc = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:",
      round(acc, 4))

# ─────────────────────────────────────────
# 7. CLASSIFICATION REPORT
# ─────────────────────────────────────────

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ─────────────────────────────────────────
# 8. CONFUSION MATRIX
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

plt.title("Confusion Matrix")
plt.show()

# ─────────────────────────────────────────
# 9. MANUAL METRICS
# ─────────────────────────────────────────

if len(np.unique(y)) == 2:

    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

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

    error_rate = 1 - accuracy

    print("\n📘 Manual Metrics")

    print("TP :", TP)
    print("TN :", TN)
    print("FP :", FP)
    print("FN :", FN)

    print("Accuracy:",
          round(accuracy, 4))

    print("Precision:",
          round(precision, 4))

    print("Recall:",
          round(recall, 4))

    print("Specificity:",
          round(specificity, 4))

    print("F1 Score:",
          round(f1, 4))

    print("Error Rate:",
          round(error_rate, 4))

# ─────────────────────────────────────────
# 10. DECISION TREE VISUALIZATION
# ─────────────────────────────────────────

plt.figure(figsize=(18, 8))

plot_tree(
    model,
    max_depth=3,
    feature_names=X.columns,
    class_names=[
        str(cls)
        for cls in np.unique(y)
    ],
    filled=True
)

plt.title("Decision Tree")
plt.show()

# ─────────────────────────────────────────
# 11. TEXT TREE
# ─────────────────────────────────────────

print("\n📘 Tree Rules\n")

print(
    export_text(
        model,
        feature_names=list(X.columns),
        max_depth=3
    )
)

# ─────────────────────────────────────────
# 12. ROC CURVE
# ─────────────────────────────────────────

if len(np.unique(y)) == 2:

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    plt.figure(figsize=(6,5))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC={round(auc,4)}"
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

    print("\nAUC ROC:",
          round(auc, 4))

# ─────────────────────────────────────────
# 13. FEATURE IMPORTANCE
# ─────────────────────────────────────────

importance_df = pd.DataFrame({

    'Feature': X.columns,

    'Importance':
    model.feature_importances_

}).sort_values(
    'Importance',
    ascending=False
)

print("\nTop Features:\n")
print(importance_df)

plt.figure(figsize=(8,5))

plt.barh(
    importance_df['Feature'][:10],
    importance_df['Importance'][:10]
)

plt.xlabel("Importance")
plt.title("Feature Importance")

plt.show()

# ─────────────────────────────────────────
# 14. OVERFITTING CHECK
# ─────────────────────────────────────────

depths = range(1, 15)

train_acc = []
test_acc = []

for d in depths:

    dt = DecisionTreeClassifier(
        max_depth=d,
        random_state=42
    )

    dt.fit(
        X_train,
        y_train
    )

    train_acc.append(
        accuracy_score(
            y_train,
            dt.predict(X_train)
        )
    )

    test_acc.append(
        accuracy_score(
            y_test,
            dt.predict(X_test)
        )
    )

plt.figure(figsize=(8,5))

plt.plot(
    depths,
    train_acc,
    marker='o',
    label='Train Accuracy'
)

plt.plot(
    depths,
    test_acc,
    marker='s',
    label='Test Accuracy'
)

plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Max Depth")
plt.legend()

plt.show()

best_depth = list(depths)[
    np.argmax(test_acc)
]

print("\nBest Depth:",
      best_depth)

# ─────────────────────────────────────────
# 15. GINI VS ENTROPY
# ─────────────────────────────────────────

print("\nGini vs Entropy")

for criterion in [
    'gini',
    'entropy'
]:

    dt = DecisionTreeClassifier(
        criterion=criterion,
        random_state=42
    )

    dt.fit(
        X_train,
        y_train
    )

    acc = accuracy_score(
        y_test,
        dt.predict(X_test)
    )

    print(
        criterion,
        "Accuracy:",
        round(acc, 4)
    )

# ─────────────────────────────────────────
# 16. CROSS VALIDATION
# ─────────────────────────────────────────

cv_acc = cross_val_score(
    DecisionTreeClassifier(
        max_depth=best_depth,
        random_state=42
    ),
    X,
    y,
    cv=10,
    scoring='accuracy'
)

print("\n10 Fold Accuracy:")

print(cv_acc.round(3))

print(
    "Mean Accuracy:",
    round(
        cv_acc.mean(),
        4
    )
)