import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
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
    StandardScaler
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

X = df.drop(
    columns=[target_column]
)

y = df[target_column]

print("\nTarget Classes:")
print(y.unique())

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

# ─────────────────────────────────────────
# 5. BUILD & TRAIN ANN
# ─────────────────────────────────────────

model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

model.fit(
    X_train,
    y_train
)

print("\n✅ ANN Trained Successfully!")

print("\nNumber of Layers:",
      model.n_layers_)

print("Iterations:",
      model.n_iter_)

print("Output Activation:",
      model.out_activation_)

# ─────────────────────────────────────────
# 6. PREDICTION
# ─────────────────────────────────────────

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)

# ─────────────────────────────────────────
# 7. ACCURACY
# ─────────────────────────────────────────

acc = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:",
      round(acc, 4))

# ─────────────────────────────────────────
# 8. CLASSIFICATION REPORT
# ─────────────────────────────────────────

print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ─────────────────────────────────────────
# 9. CONFUSION MATRIX
# ─────────────────────────────────────────

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Confusion Matrix - ANN"
)

plt.show()

# ─────────────────────────────────────────
# 10. MANUAL METRICS
# (Binary Classification Only)
# ─────────────────────────────────────────

classes = np.unique(y)

if len(classes) == 2:

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

    print("Precision:",
          round(precision,4))

    print("Recall:",
          round(recall,4))

    print("Specificity:",
          round(specificity,4))

    print("F1 Score:",
          round(f1,4))

    print("Error Rate:",
          round(error_rate,4))

# ─────────────────────────────────────────
# 11. TRAINING LOSS CURVE
# ─────────────────────────────────────────

plt.figure(figsize=(8,4))

plt.plot(
    model.loss_curve_,
    linewidth=2
)

plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.title(
    "ANN Training Loss Curve"
)

plt.show()

# ─────────────────────────────────────────
# 12. ROC CURVE
# (Binary Classification Only)
# ─────────────────────────────────────────

if len(classes) == 2:

    y_prob_binary = y_prob[:,1]

    auc = roc_auc_score(
        y_test,
        y_prob_binary
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob_binary
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

    plt.title(
        "ROC Curve - ANN"
    )

    plt.legend()

    plt.show()

    print(
        "\nAUC ROC:",
        round(auc,4)
    )

# ─────────────────────────────────────────
# 13. ARCHITECTURE COMPARISON
# ─────────────────────────────────────────

architectures = [

    (50,),
    (100,),
    (100,50),
    (100,50,25)

]

print(
    "\nArchitecture Comparison"
)

for arch in architectures:

    m = MLPClassifier(
        hidden_layer_sizes=arch,
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )

    m.fit(
        X_train,
        y_train
    )

    acc = accuracy_score(
        y_test,
        m.predict(X_test)
    )

    print(
        arch,
        "Accuracy:",
        round(acc,4)
    )

# ─────────────────────────────────────────
# 14. ACTIVATION FUNCTION COMPARISON
# ─────────────────────────────────────────

activations = [
    'relu',
    'tanh',
    'logistic'
]

print(
    "\nActivation Comparison"
)

for act in activations:

    m = MLPClassifier(
        hidden_layer_sizes=(100,50),
        activation=act,
        max_iter=500,
        random_state=42
    )

    m.fit(
        X_train,
        y_train
    )

    acc = accuracy_score(
        y_test,
        m.predict(X_test)
    )

    print(
        act,
        "Accuracy:",
        round(acc,4)
    )

# ─────────────────────────────────────────
# 15. CROSS VALIDATION
# ─────────────────────────────────────────

cv_acc = cross_val_score(

    MLPClassifier(
        hidden_layer_sizes=(100,50),
        activation='relu',
        max_iter=500,
        random_state=42
    ),

    X_scaled,
    y,

    cv=10,

    scoring='accuracy'
)

print(
    "\n10 Fold Accuracy:"
)

print(
    cv_acc.round(3)
)

print(
    "Mean Accuracy:",
    round(cv_acc.mean(),4)
)

print(
    "Std Dev:",
    round(cv_acc.std(),4)
)

# ─────────────────────────────────────────
# 16. PREDICT NEW DATA
# ─────────────────────────────────────────

print("\nEnter New Data")

new_input = []

for col in X.columns:

    value = float(
        input(
            f"Enter {col}: "
        )
    )

    new_input.append(value)

new_data = np.array([new_input])

new_scaled = scaler.transform(
    new_data
)

prediction = model.predict(
    new_scaled
)[0]

probability = model.predict_proba(
    new_scaled
)[0]

print(
    "\n✅ Predicted Class:",
    prediction
)

print("\nProbabilities")

for cls, prob in zip(
    model.classes_,
    probability
):

    print(
        f"Class {cls}: "
        f"{round(prob,4)}"
    )