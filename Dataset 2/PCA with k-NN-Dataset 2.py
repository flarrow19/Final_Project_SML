# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# Suppressing warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Loading the 'adult' dataset from OpenML
data = fetch_openml(name='adult', version=2)

# Preparing the dataset
X = data.data
y = data.target
y = y.replace({'<=50K': 0, '>50K': 1})

# Identifying categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

# Preprocessing the data: Scaling numerical features and encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X = preprocessor.fit_transform(X)
X = X.toarray()

# Splitting the dataset into training and test sets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Performing PCA and plotting the explained variance to determine the number of components
pca = PCA().fit(X)
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 4))
plt.plot(cumulative_explained_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.xticks(np.arange(0, len(cumulative_explained_variance), step=5))
plt.grid(True)
plt.show()

# Creating a PCA-KNN pipeline
pca_knn_pipeline = Pipeline([
    ('pca', PCA(n_components=50)),  # Number of components chosen based on the elbow in the plot
    ('knn', KNeighborsClassifier())
])

# Fitting the pipeline to the training data
pca_knn_pipeline.fit(X_train_scaled, y_train)

# Making predictions and evaluating the model
y_pred = pca_knn_pipeline.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
df_result = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
print(df_result)

# Hyperparameter tuning using hyperopt
def objective(params):
    pca = PCA(n_components=int(params['n_components']))
    knn = KNeighborsClassifier(n_neighbors=int(params['n_neighbors']))

    model = Pipeline([('pca', pca), ('knn', knn)])
    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, pred)
    return {'loss': -accuracy, 'status': STATUS_OK, 'accuracy': accuracy}

# Defining the space for hyperparameters
space = {
    'n_components': hp.quniform('n_components', 2, 30, 1),
    'n_neighbors': hp.quniform('n_neighbors', 1, 50, 1)
}

# Running the hyperparameter optimization
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

# Retrieve the accuracy of the best performing model
best_accuracy = max([-trial['result']['loss'] for trial in trials.trials])
print("Best hyperparameters:", best)
print("Best accuracy:", best_accuracy)


results = [{'n_components': trial['misc']['vals']['n_components'][0],
            'n_neighbors': trial['misc']['vals']['n_neighbors'][0],
            'accuracy': -trial['result']['loss']} for trial in trials.trials]

df_hyperparams = pd.DataFrame(results)

# Aggregating results in case of duplicates
aggregated_results = df_hyperparams.groupby(['n_components', 'n_neighbors'], as_index=False).mean()

# Pivoting the aggregated results to create a heatmap-ready format
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
accuracies = [-trial['result']['loss'] for trial in trials.trials]

# Plotting Classification Metric vs. Model Complexity
plt.figure(figsize=(10, 6))
sorted_indices = np.argsort(n_components_values)
sorted_n_components = np.array(n_components_values)[sorted_indices]
sorted_accuracies = np.array(accuracies)[sorted_indices]
plt.plot(sorted_n_components, sorted_accuracies, marker='o', linestyle='-', color='blue')
plt.xscale('log')
plt.xlabel('Number of Components (PCA)')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy vs. Model Complexity')
plt.show()

