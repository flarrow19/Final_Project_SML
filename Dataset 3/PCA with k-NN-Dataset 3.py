"""
Author: Kunal Malik
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import fashion_mnist
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# Suppressing warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Loading the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocessing: Flattening the images and normalizing pixel values
x_train_flattened = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test_flattened = x_test.reshape(x_test.shape[0], -1) / 255.0

# Performing PCA to understand the explained variance by components
pca = PCA().fit(x_train_flattened)
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plotting the cumulative explained variance to identify the ideal number of components
plt.figure(figsize=(8, 4))
plt.plot(cumulative_explained_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.xticks(np.arange(0, len(cumulative_explained_variance), step=50))
plt.grid(True)
plt.show()

# Creating a PCA-KNN pipeline with a chosen number of components
pca_knn_pipeline = Pipeline([
    ('pca', PCA(n_components=550)),  # Number of components based on the elbow in the plot
    ('knn', KNeighborsClassifier())
])

# Fitting the pipeline to the training data
pca_knn_pipeline.fit(x_train_flattened, y_train)

# Making predictions and evaluating the model
y_pred = pca_knn_pipeline.predict(x_test_flattened)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
df_result = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
print(df_result)

# Hyperparameter tuning using hyperopt
def objective(params):
    pca = PCA(n_components=int(params['n_components']))
    knn = KNeighborsClassifier(n_neighbors=int(params['n_neighbors']))

    model = Pipeline([('pca', pca), ('knn', knn)])

    # Using cross-validation for more robust accuracy estimates
    scores = cross_val_score(model, x_train_flattened, y_train, cv=5, scoring='accuracy')
    accuracy = scores.mean()
    return {'loss': -accuracy, 'status': STATUS_OK, 'accuracy': accuracy}

# Defining the space for hyperparameters
space = {
    'n_components': hp.quniform('n_components', 20, 100, 1),  # Range for PCA components
    'n_neighbors': hp.choice('n_neighbors', np.arange(1, 20, dtype=int))  # Range for KNN neighbors
}

# Running the hyperparameter optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,  # Number of evaluations for finding the best hyperparameters
    trials=trials
)

# Retrieve the accuracy of the best performing model
best_accuracy = max([-trial['result']['loss'] for trial in trials.trials])
print("Best hyperparameters:", best)
print("Best accuracy:", best_accuracy)

results = [{
    'n_components': int(trial['misc']['vals']['n_components'][0]),
    'n_neighbors': int(trial['misc']['vals']['n_neighbors'][0]),
    'accuracy': -trial['result']['loss']
} for trial in trials.trials]

df_results = pd.DataFrame(results)

# Aggregating results in case of duplicates by taking the mean accuracy
aggregated_results = df_results.groupby(['n_components', 'n_neighbors']).agg(np.mean).reset_index()

# Pivoting the aggregated results to create a heatmap-ready format
pivot_table = aggregated_results.pivot(index='n_components', columns='n_neighbors', values='accuracy')

# Plotting the heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Heatmap of Hyperparameter Tuning Results")
plt.xlabel("Number of Neighbors")
plt.ylabel("Number of Components")
plt.show()

train_accuracy = accuracy_score(y_train, pca_knn_pipeline.predict(x_train_flattened))

# Plotting Training vs. Test Accuracy
plt.figure(figsize=(6, 4))
plt.bar(['Train Accuracy', 'Test Accuracy'], [train_accuracy, accuracy], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Training vs. Test Accuracy')
plt.show()

# Extracting n_components, n_neighbors, and corresponding accuracies from Hyperopt trials
n_components_values = [int(trial['misc']['vals']['n_components'][0]) for trial in trials.trials]
n_neighbors_values = [int(trial['misc']['vals']['n_neighbors'][0]) for trial in trials.trials]
accuracies = [-trial['result']['loss'] for trial in trials.trials]

# Plotting Classification Metric vs. Model Complexity
plt.figure(figsize=(10, 6))
plt.scatter(n_components_values, accuracies, c=n_neighbors_values, cmap='viridis')
plt.colorbar(label='Number of Neighbors')
plt.xlabel('Number of Components (PCA)')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy vs. Model Complexity')
plt.show()
