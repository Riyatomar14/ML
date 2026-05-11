import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              roc_curve, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler, label_binarize

# 1. Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Shape:", df.shape)
print("\nClass Distribution:")
print(df['target'].value_counts())
print("Classes:", data.target_names)

# 2. Features and target
X = df.drop('target', axis=1).values
y = df['target'].values

# 3. Feature Scaling (very important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2,
    random_state=42, stratify=y)

# 5. Find best K by sweeping K from 1 to 20
k_values = range(1, 21)
train_accuracies = []
test_accuracies  = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracies.append(accuracy_score(y_train, knn.predict(X_train)))
    test_accuracies.append(accuracy_score(y_test,  knn.predict(X_test)))

# 6. Plot Accuracy vs K
plt.figure(figsize=(9, 4))
plt.plot(k_values, train_accuracies, color='steelblue',
         marker='o', linewidth=2, label='Train Accuracy')
plt.plot(k_values, test_accuracies, color='red',
         marker='s', linewidth=2, label='Test Accuracy')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K - KNN Classifier")
plt.xticks(k_values)
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_vs_k.png")
plt.show()

# 7. Best K
best_k = k_values[np.argmax(test_accuracies)]
print(f"\nBest K : {best_k}")
print(f"Best Test Accuracy : {round(max(test_accuracies), 4)}")

# 8. Train final model with best K
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

# 9. Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# 10. Accuracy
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

# 11. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=data.target_names))

# 12. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
       display_labels=data.target_names)
disp.plot(cmap='Blues')
plt.title(f"Confusion Matrix - KNN (K={best_k})")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 13. Manually calculate metrics (per class average)
TP = np.diag(cm)
FP = cm.sum(axis=0) - TP
FN = cm.sum(axis=1) - TP
TN = cm.sum() - (TP + FP + FN)

accuracy    = (TP + TN) / (TP + TN + FP + FN)
precision   = TP / (TP + FP)
recall      = TP / (TP + FN)
specificity = TN / (TN + FP)
f1          = 2 * precision * recall / (precision + recall)
error_rate  = 1 - accuracy

print("\n--- Per Class Metrics ---")
for i, cls in enumerate(data.target_names):
    print(f"\nClass: {cls}")
    print(f"  TP          : {TP[i]}")
    print(f"  TN          : {TN[i]}")
    print(f"  FP          : {FP[i]}")
    print(f"  FN          : {FN[i]}")
    print(f"  Accuracy    : {round(accuracy[i], 4)}")
    print(f"  Precision   : {round(precision[i], 4)}")
    print(f"  Recall      : {round(recall[i], 4)}")
    print(f"  Specificity : {round(specificity[i], 4)}")
    print(f"  F1-Score    : {round(f1[i], 4)}")
    print(f"  Error Rate  : {round(error_rate[i], 4)}")

# 14. AUC-ROC (multi-class)
y_bin = label_binarize(y_test, classes=[0, 1, 2])
auc   = roc_auc_score(y_bin, y_prob,
                       multi_class='ovr', average='macro')
print(f"\nAUC-ROC (macro avg): {round(auc, 4)}")

# 15. ROC Curve for each class
plt.figure(figsize=(7, 5))
colors = ['steelblue', 'red', 'green']
for i, (cls, color) in enumerate(zip(data.target_names, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    auc_i = roc_auc_score(y_bin[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, color=color, linewidth=2,
             label=f'{cls} (AUC={round(auc_i, 3)})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - KNN (K={best_k})")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# 16. Effect of distance metric
metrics = ['euclidean', 'manhattan', 'minkowski']
print("\n--- Distance Metric Comparison ---")
for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=best_k,
                                metric=metric)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    print(f"  {metric:12s} -> Accuracy: {round(acc, 4)}")

# 17. Cross Validation (10-fold)
cv_acc = cross_val_score(
    KNeighborsClassifier(n_neighbors=best_k),
    X_scaled, y, cv=10, scoring='accuracy')

cv_auc = cross_val_score(
    KNeighborsClassifier(n_neighbors=best_k),
    X_scaled, y, cv=10, scoring='roc_auc_ovr')

print("\n10-Fold CV Accuracy:", cv_acc.round(3))
print("Mean Accuracy :", round(cv_acc.mean(), 4))
print("Std Dev       :", round(cv_acc.std(), 4))
print("\n10-Fold CV AUC:", cv_auc.round(3))
print("Mean AUC      :", round(cv_auc.mean(), 4))