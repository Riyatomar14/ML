import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.preprocessing import label_binarize

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)
print("Shape:", X.shape)

# 2. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train the model
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# 5. Accuracy
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

# 6. Classification Report (Precision, Recall, F1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 8. AUC-ROC Score (multi-class)
y_bin = label_binarize(y_test, classes=[0, 1, 2])
auc = roc_auc_score(y_bin, y_prob, multi_class='ovr')
print("\nAUC-ROC Score:", round(auc, 4))

# 9. 10-Fold Cross Validation
cv_scores = cross_val_score(GaussianNB(), X, y, cv=10, scoring='accuracy')
print("\n10-Fold CV Scores:", cv_scores.round(3))
print("Mean Accuracy:", round(cv_scores.mean(), 4))
print("Std Dev:", round(cv_scores.std(), 4))