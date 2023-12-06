#!/usr/bin/env python
# coding: utf-8

# Importing required libraries
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import warnings
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import seaborn as sns
import matplotlib.pyplot as plt

# Suppressing warnings for cleaner output
warnings.filterwarnings("ignore")

# Loading the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocessing the data
X_train_scaled = x_train.reshape((-1, 28, 28, 1)) / 255.0
X_test_scaled = x_test.reshape((-1, 28, 28, 1)) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Defining the CNN model architecture
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compiling the CNN model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the CNN model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32)

# Making predictions with the CNN model
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculating the accuracy of the CNN model
accuracy = accuracy_score(y_true, y_pred_classes)
print(f'Accuracy: {accuracy:.4f}')

# Generating a classification report
result = classification_report(y_true, y_pred_classes, output_dict=True)

# Converting the classification report into a DataFrame
df_result = pd.DataFrame(result).transpose()

# Defining a function to create a CNN model with given hyperparameters
def create_model(params):
    model = Sequential([
        Conv2D(params['filters'], kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(params['dropout_conv']),
        Flatten(),
        Dense(params['dense_units'], activation='relu'),
        Dropout(params['dropout_dense']),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Defining the objective function for hyperparameter tuning
def objective(params):
    model = create_model(params)
    model.fit(X_train_scaled, y_train, epochs=5, batch_size=128, verbose=0)
    _, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    return {'loss': -accuracy, 'status': STATUS_OK}

# Defining the space for hyperparameter optimization
space = {
    'filters': hp.choice('filters', [32, 64, 128]),
    'dropout_conv': hp.uniform('dropout_conv', 0.1, 0.5),
    'dense_units': hp.choice('dense_units', [64, 128, 256]),
    'dropout_dense': hp.uniform('dropout_dense', 0.1, 0.5)
}

# Running the hyperparameter optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=20,
    trials=trials
)

# Retrieving the best accuracy from the trials
best_accuracy = max([trial['result']['loss'] for trial in trials.trials])
print("Best Hyperparameters: ", best)
print("Best Accuracy: ", best_accuracy)

# Collect the hyperparameter combinations and their corresponding accuracies
results = []
for trial in trials.trials:
    vals = trial['misc']['vals']
    # Note that we are using the "vals" key to extract hyperparameter values
    results.append({
        'filters': vals['filters'][0] if vals['filters'] else None,
        'dropout_conv': vals['dropout_conv'][0] if vals['dropout_conv'] else None,
        'dense_units': vals['dense_units'][0] if vals['dense_units'] else None,
        'dropout_dense': vals['dropout_dense'][0] if vals['dropout_dense'] else None,
        'accuracy': -trial['result']['loss']
    })

# Convert the list of results to a DataFrame
df_results = pd.DataFrame(results)

# Assuming 'filters', 'dense_units' are categorical hyperparameters we convert them back
# to their original categorical values if necessary
df_results['filters'] = df_results['filters'].map({0: 32, 1: 64, 2: 128})
df_results['dense_units'] = df_results['dense_units'].map({0: 64, 1: 128, 2: 256})

# Aggregate the results by taking the mean accuracy for each combination of hyperparameters
# We are assuming that 'filters' and 'dense_units' are the hyperparameters we want to pivot on
aggregated_results = df_results.groupby(['filters', 'dense_units'], as_index=False).mean()

# Pivoting the aggregated results to create a format suitable for a heatmap
pivot_table = aggregated_results.pivot(index='filters', columns='dense_units', values='accuracy')

# Plotting the heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Heatmap of Hyperparameter Tuning Results")
plt.xlabel("Dense Units")
plt.ylabel("Filters")
plt.show()

# Modify the model training to capture history
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=0)

# Extracting loss values
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# Plotting Train vs Test Loss as a bar graph
plt.figure(figsize=(10, 6))
plt.bar(epochs, train_loss, alpha=0.6, label='Train Loss', color='blue')
plt.bar(epochs, val_loss, alpha=0.6, label='Validation Loss', color='red')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Define a range of filters to test
filter_range = [32, 64, 128]
accuracies = []

for filters in filter_range:
    # Create and train the model
    model = create_model({'filters': filters, 'dropout_conv': 0.3, 'dense_units': 128, 'dropout_dense': 0.4})
    model.fit(X_train_scaled, y_train, epochs=5, batch_size=128, verbose=0)
    _, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    accuracies.append(accuracy)

# Plotting Accuracy vs. Number of Filters
plt.figure(figsize=(10, 6))
plt.plot(filter_range, accuracies, marker='o')
plt.title('Classification Accuracy vs. Number of Filters')
plt.xlabel('Number of Filters')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
