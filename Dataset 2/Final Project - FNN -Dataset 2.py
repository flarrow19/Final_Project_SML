#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import legacy
from keras.optimizers import legacy as optimizers
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.optimizers import Adam
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Suppressing warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Loading the Adult dataset from OpenML
data = fetch_openml(name='adult', version=2)

# Preparing the dataset for model training
X = data.data
y = data.target.replace({'<=50K': 0, '>50K': 1})

# Identifying categorical and numerical columns for preprocessing
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

# Creating a preprocessing pipeline with StandardScaler and OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Applying the preprocessing pipeline to feature data
X = preprocessor.fit_transform(X)
X = X.toarray()  # Converting to array for neural network compatibility

# Splitting the data into training and testing sets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Building the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Training the model with the training data
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32)

# Making predictions with the test data
y_pred = model.predict(X_test_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy:.4f}')

# Generating a classification report and converting it to a DataFrame
result = classification_report(y_test, y_pred_binary, output_dict=True)
df_result = pd.DataFrame(result).transpose()

# Define a function to create a neural network model with given hyperparameters
def create_model(params):
    model = Sequential()
    model.add(Dense(units=int(params['units']), activation=params['activation'], input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define an objective function for hyperparameter tuning
def objective(params):
    model = create_model(params)
    model.fit(X_train_scaled, y_train, epochs=int(params['epochs']), batch_size=int(params['batch_size']), verbose=0)
    y_pred = model.predict(X_test_scaled).round()
    accuracy = accuracy_score(y_test, y_pred)
    return {'loss': -accuracy, 'status': STATUS_OK, 'accuracy': accuracy}

# Defining the space for hyperparameter optimization
space = {
    'units': hp.quniform('units', 32, 256, 1),
    'activation': hp.choice('activation', ['relu', 'tanh']),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'epochs': hp.quniform('epochs', 10, 50, 1),
    'batch_size': hp.quniform('batch_size', 16, 128, 1)
}

# Running the hyperparameter optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

# Retrieving the best accuracy from the trials
best_accuracy = max([-trial['result']['loss'] for trial in trials.trials])
print("Best hyperparameters:", best)
print("Best accuracy:", best_accuracy)

# Extracting the scores and hyperparameters from the trials
results = [{
    'units': trial['misc']['vals']['units'][0],
    'batch_size': trial['misc']['vals']['batch_size'][0],
    'score': -trial['result']['loss']
} for trial in trials.trials]

# Creating a DataFrame from the extracted scores and parameters
df_results = pd.DataFrame(results)

# Aggregate results by taking the mean of scores for each combination of hyperparameters
aggregated_results = df_results.groupby(['units', 'batch_size']).agg('mean').reset_index()

# Pivoting the aggregated results to create a format suitable for a heatmap
pivot_table = aggregated_results.pivot(index='units', columns='batch_size', values='score')

# Plotting the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title('Heatmap of Hyperparameter Tuning Results')
plt.xlabel('Batch Size')
plt.ylabel('Units')
plt.show()

history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=0)
# Extracting loss values
train_loss = history.history['loss']
test_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# Plotting Train vs Test Loss
plt.figure(figsize=(10, 6))
plt.bar(epochs, train_loss, alpha=0.6, label='Train Loss')
plt.bar(epochs, test_loss, alpha=0.6, label='Test Loss', color='r')
plt.title('Training vs. Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

units_range = range(32, 257, 32)
accuracies = []

for units in units_range:
    model = create_model({'units': units, 'activation': 'relu', 'learning_rate': 0.001, 'epochs': 10, 'batch_size': 32})
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)
    y_pred = model.predict(X_test_scaled).round()
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plotting Accuracy vs Number of Units
plt.figure(figsize=(10, 6))
plt.plot(units_range, accuracies, marker='o')
plt.title('Classification Accuracy vs. Model Complexity (Number of Units)')
plt.xlabel('Number of Units')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


