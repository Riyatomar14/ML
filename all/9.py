import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              roc_curve, ConfusionMatrixDisplay)

# ── Save folder: PNGs saved in the same folder as this script ─────────────
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

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

# 3. Split data (no scaling needed for Decision Tree)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y)

# 4. Train default Decision Tree
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)

print("\nTree Depth       :", model.get_depth())
print("Number of Leaves :", model.get_n_leaves())

# 5. Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. Accuracy
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

# 7. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Malignant', 'Benign']))

# 8. Confusion Matrix with all metrics
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

# 9. Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
       display_labels=['Malignant', 'Benign'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))  # FIXED
plt.show()

# 10. Visualize the Tree (limited depth for clarity)
plt.figure(figsize=(20, 8))
plot_tree(model,
          max_depth=3,
          feature_names=data.feature_names,
          class_names=['Malignant', 'Benign'],
          filled=True,
          rounded=True,
          fontsize=8)
plt.title("Decision Tree (max_depth=3 shown)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "decision_tree.png"), dpi=150)  # FIXED
plt.show()

# 11. Print text representation
print("\nText Tree (depth 3):")
print(export_text(model,
      feature_names=list(data.feature_names),
      max_depth=3))

# 12. ROC Curve
auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='steelblue',
         linewidth=2, label=f'AUC = {round(auc, 4)}')
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "roc_curve.png"))  # FIXED
plt.show()

print("\nAUC-ROC Score:", round(auc, 4))

# 13. Feature Importance
feat_df = pd.DataFrame({
    'Feature'   : data.feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(data=feat_df.head(10),
            x='Importance', y='Feature',
            palette='coolwarm')
plt.title("Top 10 Feature Importances - Decision Tree")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "feature_importance.png"))  # FIXED
plt.show()

print("\nTop 10 Features:")
print(feat_df.head(10).to_string(index=False))

# 14. Effect of max_depth - Overfitting check
depths = range(1, 15)
train_accs = []
test_accs  = []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, dt.predict(X_train)))
    test_accs.append(accuracy_score(y_test, dt.predict(X_test)))

plt.figure(figsize=(9, 4))
plt.plot(depths, train_accs, color='steelblue',
         marker='o', linewidth=2, label='Train Accuracy')
plt.plot(depths, test_accs, color='red',
         marker='s', linewidth=2, label='Test Accuracy')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Max Depth - Decision Tree")
plt.xticks(depths)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "depth_vs_accuracy.png"))  # FIXED
plt.show()

best_depth = list(depths)[np.argmax(test_accs)]  # FIXED: range -> list
print(f"\nBest max_depth     : {best_depth}")
print(f"Best Test Accuracy : {round(max(test_accs), 4)}")

# 15. Compare Gini vs Entropy
print("\n--- Gini vs Entropy ---")
for criterion in ['gini', 'entropy']:
    dt = DecisionTreeClassifier(criterion=criterion, random_state=42)
    dt.fit(X_train, y_train)
    acc = accuracy_score(y_test, dt.predict(X_test))
    print(f"  {criterion:8s} -> Accuracy: {round(acc, 4)}"
          f" | Depth: {dt.get_depth()}"
          f" | Leaves: {dt.get_n_leaves()}")

# 16. Best model with tuned depth
best_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=best_depth,
    random_state=42
)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)
print(f"\nTuned Model Accuracy (depth={best_depth}):",
      round(accuracy_score(y_test, y_pred_best), 4))

# 17. Cross Validation (10-fold)
cv_acc = cross_val_score(
    DecisionTreeClassifier(max_depth=best_depth, random_state=42),
    X, y, cv=10, scoring='accuracy')

cv_auc = cross_val_score(
    DecisionTreeClassifier(max_depth=best_depth, random_state=42),
    X, y, cv=10, scoring='roc_auc')

print("\n10-Fold CV Accuracy:", cv_acc.round(3))
print("Mean Accuracy :", round(cv_acc.mean(), 4))
print("Std Dev       :", round(cv_acc.std(), 4))
print("\n10-Fold CV AUC:", cv_auc.round(3))
print("Mean AUC      :", round(cv_auc.mean(), 4))