#!/usr/bin/env python
# coding: utf-8
"""
Author: Shubham Gade
"""

# Importing necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
# Loading the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Display feature and target names for understanding the dataset
print(data.feature_names)
print(data.target_names)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Performing Principal Component Analysis (PCA)
X_scaled = StandardScaler().fit_transform(X)
pca = PCA().fit(X_scaled)
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plotting the explained variance to determine the number of components
plt.figure(figsize=(8, 4))
plt.plot(cumulative_explained_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.xticks(np.arange(0, len(cumulative_explained_variance), step=1))
plt.grid(True)
plt.show()

# Creating a PCA-KNN pipeline
pca_knn_pipeline = Pipeline([
    ('pca', PCA(n_components=12)),  # Chosen based on the elbow curve
    ('knn', KNeighborsClassifier())
])
pca_knn_pipeline.fit(X_train_scaled, y_train)

# Making predictions and evaluating the model
y_pred = pca_knn_pipeline.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
df_result = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
print(df_result)

# Hyperparameter tuning using hyperopt
hyperparam_results = []

def objective(params):
    n_components = int(params['n_components'])
    n_neighbors = int(params['n_neighbors'])
    pca = PCA(n_components=n_components)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    model = Pipeline([('pca', pca), ('knn', knn)])
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, pred)
    hyperparam_results.append({'n_components': n_components, 'n_neighbors': n_neighbors, 'accuracy': accuracy})
    return {'loss': -accuracy, 'status': STATUS_OK}

# Defining the space for hyperparameters
space = {
    'n_components': hp.quniform('n_components', 2, 30, 1),
    'n_neighbors': hp.quniform('n_neighbors', 1, 50, 1)
}

# Running the hyperparameter optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
best_accuracy = max([-trial['result']['loss'] for trial in trials.trials])
print("Best hyperparameters:", best)
print("Best accuracy:", best_accuracy)

df_hyperparams = pd.DataFrame(hyperparam_results)


aggregated_results = df_hyperparams.groupby(['n_components', 'n_neighbors']).agg('mean').reset_index()


pivot_table = aggregated_results.pivot(index='n_components', columns='n_neighbors', values='accuracy')

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Heatmap of Hyperparameter Tuning Results")
plt.xlabel("Number of Neighbors")
plt.ylabel("Number of Components")
plt.show()

train_accuracy = accuracy_score(y_train, pca_knn_pipeline.predict(X_train_scaled))
test_accuracy = accuracy_score(y_test, y_pred)

# Plotting Training vs. Test Accuracy
plt.figure(figsize=(6, 4))
plt.bar(['Train Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Training vs. Test Accuracy')
plt.show()

# Extracting n_components, n_neighbors, and corresponding accuracies from Hyperopt trials
n_components_values = [trial['misc']['vals']['n_components'][0] for trial in trials.trials]
n_neighbors_values = [trial['misc']['vals']['n_neighbors'][0] for trial in trials.trials]
accuracies = [trial['result']['accuracy'] for trial in trials.trials]

# Plotting Classification Metric vs. Model Complexity
plt.figure(figsize=(10, 6))
plt.scatter(n_components_values, accuracies, c=n_neighbors_values, cmap='viridis')
plt.colorbar(label='Number of Neighbors')
plt.xscale('log')
plt.xlabel('Number of Components (PCA)')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy vs. Model Complexity')
plt.show()
