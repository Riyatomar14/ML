import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              roc_curve, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Shape:", df.shape)
print("\nClass Distribution:")
print(df['target'].value_counts())
print("0 = Malignant, 1 = Benign")

# 2. Features and target
X = df.drop('target', axis=1).values
y = df['target'].values

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2,
    random_state=42, stratify=y)

# 5. Build and Train ANN
model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)
model.fit(X_train, y_train)

print("\nNumber of layers    :", model.n_layers_)
print("Number of iterations:", model.n_iter_)
print("Output activation   :", model.out_activation_)

# 6. Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 7. Accuracy
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

# 8. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Malignant', 'Benign']))

# 9. Confusion Matrix with all metrics
cm = confusion_matrix(y_test, y_pred)
TP = cm[1][1]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]

accuracy    = (TP + TN) / (TP + TN + FP + FN)
precision   = TP / (TP + FP)
recall      = TP / (TP + FN)
specificity = TN / (TN + FP)
f1          = 2 * precision * recall / (precision + recall)
error_rate  = 1 - accuracy

print("\n--- Manually Calculated Metrics ---")
print(f"TP          : {TP}")
print(f"TN          : {TN}")
print(f"FP          : {FP}")
print(f"FN          : {FN}")
print(f"Accuracy    : {round(accuracy, 4)}")
print(f"Precision   : {round(precision, 4)}")
print(f"Recall      : {round(recall, 4)}")
print(f"Specificity : {round(specificity, 4)}")
print(f"F1-Score    : {round(f1, 4)}")
print(f"Error Rate  : {round(error_rate, 4)}")

# 10. Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
       display_labels=['Malignant', 'Benign'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - ANN")
plt.tight_layout()
plt.savefig("ann_confusion_matrix.png")
plt.show()

# 11. Training Loss Curve (FIXED)
plt.figure(figsize=(8, 4))
plt.plot(model.loss_curve_, color='steelblue',
         linewidth=2, label='Training Loss')
if hasattr(model, 'best_loss_') and model.best_loss_ is not None:
    plt.axhline(y=model.best_loss_, color='red',
                linestyle='--', label='Best Loss')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("ANN Training Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

# 12. ROC Curve
auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='steelblue',
         linewidth=2, label=f'AUC = {round(auc, 4)}')
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - ANN")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print("\nAUC-ROC Score:", round(auc, 4))

# 13. Compare different architectures
architectures = [
    (50,),
    (100,),
    (100, 50),
    (100, 50, 25),
    (200, 100, 50)
]

print("\n--- Architecture Comparison ---")
for arch in architectures:
    m = MLPClassifier(hidden_layer_sizes=arch,
                      activation='relu',
                      solver='adam',
                      max_iter=500,
                      random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    print(f"  {str(arch):20s} -> Accuracy: {round(acc, 4)}")

# 14. Compare activation functions
activations = ['relu', 'tanh', 'logistic']
print("\n--- Activation Function Comparison ---")
for act in activations:
    m = MLPClassifier(hidden_layer_sizes=(100, 50),
                      activation=act,
                      solver='adam',
                      max_iter=500,
                      random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    print(f"  {act:10s} -> Accuracy: {round(acc, 4)}")

# 15. Cross Validation (10-fold)
cv_acc = cross_val_score(
    MLPClassifier(hidden_layer_sizes=(100, 50),
                  activation='relu',
                  solver='adam',
                  max_iter=500,
                  random_state=42),
    X_scaled, y, cv=10, scoring='accuracy')

cv_auc = cross_val_score(
    MLPClassifier(hidden_layer_sizes=(100, 50),
                  activation='relu',
                  solver='adam',
                  max_iter=500,
                  random_state=42),
    X_scaled, y, cv=10, scoring='roc_auc')

print("\n10-Fold CV Accuracy:", cv_acc.round(3))
print("Mean Accuracy :", round(cv_acc.mean(), 4))
print("Std Dev       :", round(cv_acc.std(), 4))
print("\n10-Fold CV AUC:", cv_auc.round(3))
print("Mean AUC      :", round(cv_auc.mean(), 4))