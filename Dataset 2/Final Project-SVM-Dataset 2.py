#!/usr/bin/env python
# coding: utf-8

# Importing required libraries
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Fetching the 'adult' dataset
data = fetch_openml(name='adult', version=1)
X = data.data
y = data.target

# Identifying numerical and categorical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Creating a preprocessor for scaling and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Applying transformations to the features
X = preprocessor.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training different SVM models with various kernels
svm_model_linear = SVC(kernel='linear')
svm_model_linear.fit(X_train_scaled, y_train)

svm_model_radial = SVC(kernel='rbf', gamma='scale')
svm_model_radial.fit(X_train_scaled, y_train)

svm_model_sigmoid = SVC(kernel='sigmoid', gamma='scale', coef0=0)
svm_model_sigmoid.fit(X_train_scaled, y_train)

svm_model = SVC(kernel='poly', degree=3, coef0=1)
svm_model.fit(X_train_scaled, y_train)

# Making predictions with the trained models
y_pred = svm_model.predict(X_test_scaled)
y_pred_linear = svm_model_linear.predict(X_test_scaled)
y_pred_radial = svm_model_radial.predict(X_test_scaled)
y_pred_sigmoid = svm_model_sigmoid.predict(X_test_scaled)

# Printing accuracy scores for each kernel type
print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred_linear))
print(accuracy_score(y_test, y_pred_sigmoid))
print(accuracy_score(y_test, y_pred_radial))

# Generating a classification report
result = classification_report(y_test, y_pred, output_dict=True)
df_result = pd.DataFrame(result).transpose()

# Performing grid search for hyperparameter tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'degree': [2, 3, 4, 5],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
    'coef0': [0.0, 0.1, 0.5, 1]
}

# Initializing GridSearchCV with the SVM model
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=3)
grid_search.fit(X_train_scaled, y_train)

# Extracting the best parameters and calculating accuracy
best_params = grid_search.best_params_
print(best_params)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy*100)

# Generating a heatmap from the grid search results
scores_matrix = grid_search.cv_results_['mean_test_score'].reshape(len(param_grid['C']), len(param_grid['degree']), len(param_grid['gamma']), len(param_grid['coef0']))

# Plotting the heatmap for one slice of hyperparameters for demonstration
plt.figure(figsize=(10, 8))
sns.heatmap(scores_matrix[:, :, 0, 0], annot=True, cmap='YlGnBu', xticklabels=param_grid['degree'], yticklabels=param_grid['C'])
plt.title('Grid Search Accuracy Scores Heatmap')
plt.xlabel('Degree')
plt.ylabel('C')
plt.show()

train_losses = []
test_losses = []
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

for C in C_values:
    model = SVC(C=C, kernel='linear', probability=True)
    model.fit(X_train_scaled, y_train)
    train_probs = model.predict_proba(X_train_scaled)
    test_probs = model.predict_proba(X_test_scaled)
    train_loss = log_loss(y_train, train_probs)
    test_loss = log_loss(y_test, test_probs)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

plt.figure(figsize=(10, 8))
plt.plot(C_values, train_losses, label='Train Loss')
plt.plot(C_values, test_losses, label='Test Loss')
plt.xscale('log')
plt.xlabel('C (Model Complexity)')
plt.ylabel('Log Loss')
plt.title('Training vs. Test Loss for Different C Values')
plt.legend()
plt.show()

# Example of plotting classification metric vs. model complexity
# For this, we can use the accuracy results from GridSearchCV across different 'C' values
# Assuming 'C' represents model complexity and using the first 'gamma' and 'coef0' value for simplicity
mean_test_scores = grid_search.cv_results_['mean_test_score'].reshape(len(param_grid['C']), len(param_grid['degree']), len(param_grid['gamma']), len(param_grid['coef0']))[:, 0, 0, 0]

plt.figure(figsize=(10, 8))
plt.plot(param_grid['C'], mean_test_scores, marker='o')
plt.xscale('log')
plt.xlabel('C (Model Complexity)')
plt.ylabel('Mean Test Accuracy')
plt.title('Classification Accuracy vs. Model Complexity')
plt.grid(True)
plt.show()


