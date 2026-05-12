import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      GridSearchCV)
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

# 3. Feature Scaling (very important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2,
    random_state=42, stratify=y)

# 5. Train SVM with RBF kernel
model = SVC(kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42)
model.fit(X_train, y_train)

print("\nNumber of Support Vectors:", model.n_support_)
print("Total Support Vectors    :", sum(model.n_support_))

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
plt.title("Confusion Matrix - SVM")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 11. ROC Curve
auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='steelblue',
         linewidth=2, label=f'AUC = {round(auc, 4)}')
plt.plot([0, 1], [0, 1], 'r--',
         label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print("\nAUC-ROC Score:", round(auc, 4))

# 12. Compare different kernels
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
print("\n--- Kernel Comparison ---")
kernel_results = []
for kernel in kernels:
    svm = SVC(kernel=kernel,
              probability=True,
              random_state=42)
    svm.fit(X_train, y_train)
    y_p  = svm.predict(X_test)
    y_pr = svm.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, y_p)
    auc_k = roc_auc_score(y_test, y_pr)
    kernel_results.append({
        'Kernel'  : kernel,
        'Accuracy': round(acc, 4),
        'AUC'     : round(auc_k, 4)
    })
    print(f"  {kernel:8s} -> Accuracy: {round(acc,4)}"
          f"  AUC: {round(auc_k,4)}")

# 13. Effect of C parameter
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_accs = []
test_accs  = []

for C in C_values:
    svm = SVC(kernel='rbf', C=C,
              gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train,
                      svm.predict(X_train)))
    test_accs.append(accuracy_score(y_test,
                     svm.predict(X_test)))

plt.figure(figsize=(9, 4))
plt.plot(range(len(C_values)), train_accs,
         color='steelblue', marker='o',
         linewidth=2, label='Train Accuracy')
plt.plot(range(len(C_values)), test_accs,
         color='red', marker='s',
         linewidth=2, label='Test Accuracy')
plt.xticks(range(len(C_values)),
           [str(c) for c in C_values])
plt.xlabel("C Value")
plt.ylabel("Accuracy")
plt.title("Effect of C on SVM Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("c_vs_accuracy.png")
plt.show()

# 14. Effect of gamma parameter
gamma_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]
train_accs_g = []
test_accs_g  = []

for g in gamma_values:
    svm = SVC(kernel='rbf', C=1.0,
              gamma=g, random_state=42)
    svm.fit(X_train, y_train)
    train_accs_g.append(accuracy_score(y_train,
                        svm.predict(X_train)))
    test_accs_g.append(accuracy_score(y_test,
                       svm.predict(X_test)))

plt.figure(figsize=(9, 4))
plt.plot(range(len(gamma_values)), train_accs_g,
         color='steelblue', marker='o',
         linewidth=2, label='Train Accuracy')
plt.plot(range(len(gamma_values)), test_accs_g,
         color='red', marker='s',
         linewidth=2, label='Test Accuracy')
plt.xticks(range(len(gamma_values)),
           [str(g) for g in gamma_values])
plt.xlabel("Gamma Value")
plt.ylabel("Accuracy")
plt.title("Effect of Gamma on SVM Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("gamma_vs_accuracy.png")
plt.show()

# 15. GridSearchCV - best C and gamma
print("\n--- GridSearchCV for best C and gamma ---")
param_grid = {
    'C'    : [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(probability=True,
                        random_state=42),
                    param_grid,
                    cv=5,
                    scoring='accuracy',
                    verbose=1)
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", round(grid.best_score_, 4))

# 16. Final model with best params
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

print("\nTuned Model Accuracy:",
      round(accuracy_score(y_test, y_pred_best), 4))
print("Tuned Model AUC     :",
      round(roc_auc_score(y_test, y_prob_best), 4))

# 17. Cross Validation (10-fold)
cv_acc = cross_val_score(
    SVC(kernel='rbf',
        C=grid.best_params_['C'],
        gamma=grid.best_params_['gamma'],
        probability=True,
        random_state=42),
    X_scaled, y, cv=10, scoring='accuracy')

cv_auc = cross_val_score(
    SVC(kernel='rbf',
        C=grid.best_params_['C'],
        gamma=grid.best_params_['gamma'],
        probability=True,
        random_state=42),
    X_scaled, y, cv=10, scoring='roc_auc')

print("\n10-Fold CV Accuracy:", cv_acc.round(3))
print("Mean Accuracy :", round(cv_acc.mean(), 4))
print("Std Dev       :", round(cv_acc.std(), 4))
print("\n10-Fold CV AUC:", cv_auc.round(3))
print("Mean AUC      :", round(cv_auc.mean(), 4))