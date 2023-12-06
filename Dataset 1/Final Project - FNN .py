# Import necessary libraries
import pandas as pd
import numpy as np
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Data normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Define the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32)

# Model evaluation
y_pred = model.predict(X_test_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy:.4f}')
df_result = pd.DataFrame(classification_report(y_test, y_pred_binary, output_dict = True)).transpose()

# Hyperparameter tuning using hyperopt
def create_model(params):
    # Model creation for hyperparameter tuning
    model = Sequential()
    model.add(Dense(units=int(params['units']), activation=params['activation'], input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def objective(params):
    # Objective function for hyperopt
    model = create_model(params)
    model.fit(X_train_scaled, y_train, epochs=int(params['epochs']), batch_size=int(params['batch_size']), verbose=0)
    y_pred = model.predict(X_test_scaled).round()
    accuracy = accuracy_score(y_test, y_pred)
    return {'loss': -accuracy, 'status': STATUS_OK, 'accuracy': accuracy}

# Define the search space for hyperopt
space = {
    'units': hp.quniform('units', 32, 256, 1),
    'activation': hp.choice('activation', ['relu', 'tanh']),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'epochs': hp.quniform('epochs', 10, 50, 1),
    'batch_size': hp.quniform('batch_size', 16, 128, 1)
}

# Running hyperopt
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

best_accuracy = max([-trial['result']['loss'] for trial in trials.trials])
print("Best hyperparameters:", best)
print("Best accuracy:", best_accuracy)

# Model evaluation
y_pred = model.predict(X_test_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy:.4f}')
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Train the model and save history
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=0)

# Plotting training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Initialize lists to store complexities and accuracies
complexities = []
accuracies = []

# Loop over different complexities (e.g., number of neurons)
for units in range(10, 100, 10):
    model = Sequential()
    model.add(Dense(units, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)
    _, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    complexities.append(units)
    accuracies.append(accuracy)

# Plotting accuracy vs. model complexity
plt.figure(figsize=(10, 6))
plt.plot(complexities, accuracies)
plt.title('Model Accuracy vs. Complexity')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neurons in the Hidden Layer')
plt.show()

