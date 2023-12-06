#!/usr/bin/env python
"""
Author: Kunal Malik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model with increased max_iter to ensure convergence
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Calculate and print the accuracy and classification report
accuracy = accuracy_score(y_test, y_pred) * 100
print(f'Accuracy: {accuracy}%')
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix of Logistic Regression Model')
plt.show()

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l2'],  # Based on the convergence warning, using only 'l2'
    'solver': ['liblinear']  # Using 'liblinear' as it converges for 'l1' and 'l2'
}
grid_search = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best hyperparameters and accuracy
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
best_accuracy = accuracy_score(y_test, best_model.predict(X_test_scaled)) * 100
print(f'Best Hyperparameters: {best_params}')
print(f'Best Accuracy: {best_accuracy}%')

train_loss = log_loss(y_train, model.predict_proba(X_train_scaled))
test_loss = log_loss(y_test, model.predict_proba(X_test_scaled))

labels = ['Train Loss', 'Test Loss']
loss_values = [train_loss, test_loss]

plt.figure(figsize=(8, 6))
plt.bar(labels, loss_values, color=['blue', 'orange'])
plt.title('Training vs. Test Loss')
plt.ylabel('Log Loss')
for i, v in enumerate(loss_values):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
plt.show()

# 2. Plot of Classification Metric vs. Model Complexity
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_accuracies = []
test_accuracies = []

for C in C_values:
    model = LogisticRegression(C=C, max_iter=10000, penalty='l2', solver='liblinear')
    model.fit(X_train_scaled, y_train)
    train_accuracies.append(accuracy_score(y_train, model.predict(X_train_scaled)))
    test_accuracies.append(accuracy_score(y_test, model.predict(X_test_scaled)))

plt.figure()
plt.semilogx(C_values, train_accuracies, label='Train Accuracy')
plt.semilogx(C_values, test_accuracies, label='Test Accuracy')
plt.title('Accuracy vs. Model Complexity (Regularization Strength)')
plt.xlabel('C (Regularization Strength)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
