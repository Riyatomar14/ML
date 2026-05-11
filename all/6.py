import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
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

# 2. Check missing values
print("\nMissing values:", df.isnull().sum().sum())

# 3. Visualize class distribution
plt.figure(figsize=(5, 3))
sns.countplot(x='target', data=df,
              palette=['salmon', 'steelblue'])
plt.xticks([0, 1], ['Malignant', 'Benign'])
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

# 4. Features and target
X = df.drop('target', axis=1).values
y = df['target'].values

# 5. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2,
    random_state=42, stratify=y)

# 7. Train the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 8. Predict
y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)[:, 1]

# 9. Accuracy
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

# 10. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Malignant', 'Benign']))

# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
TP = cm[1][1]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]

print("\n--- Confusion Matrix Values ---")
print(f"TP (True Positive)  : {TP}")
print(f"TN (True Negative)  : {TN}")
print(f"FP (False Positive) : {FP}")
print(f"FN (False Negative) : {FN}")

# Manual metric calculation
accuracy    = (TP + TN) / (TP + TN + FP + FN)
precision   = TP / (TP + FP)
recall      = TP / (TP + FN)       # Sensitivity
specificity = TN / (TN + FP)
f1          = 2 * precision * recall / (precision + recall)
error_rate  = 1 - accuracy

print("\n--- Manually Calculated Metrics ---")
print("Accuracy    :", round(accuracy, 4))
print("Precision   :", round(precision, 4))
print("Recall      :", round(recall, 4))
print("Specificity :", round(specificity, 4))
print("F1-Score    :", round(f1, 4))
print("Error Rate  :", round(error_rate, 4))

# 12. Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
       display_labels=['Malignant', 'Benign'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 13. AUC - ROC Curve
auc = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='steelblue',
         linewidth=2, label=f'AUC = {round(auc, 4)}')
plt.plot([0, 1], [0, 1], color='red',
         linestyle='--', label='Random Classifier')
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print("\nAUC-ROC Score:", round(auc, 4))

# 14. Feature Importance
coef_df = pd.DataFrame({
    'Feature'    : data.feature_names,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(data=coef_df.head(10),
            x='Coefficient', y='Feature',
            palette='coolwarm')
plt.title("Top 10 Most Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# 15. Cross Validation (10-fold)
cv_acc = cross_val_score(
    LogisticRegression(max_iter=1000),
    X_scaled, y, cv=10, scoring='accuracy')
cv_auc = cross_val_score(
    LogisticRegression(max_iter=1000),
    X_scaled, y, cv=10, scoring='roc_auc')

print("\n10-Fold CV Accuracy:", cv_acc.round(3))
print("Mean Accuracy:", round(cv_acc.mean(), 4))
print("\n10-Fold CV AUC:", cv_auc.round(3))
print("Mean AUC     :", round(cv_auc.mean(), 4))