"""
Author: Shubham Gade
"""
# Importing required libraries
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the image data
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encoding the labels
encoder = OneHotEncoder(sparse=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

# Creating the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(10, activation='softmax')  # Output layer with 10 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training the model with validation data and saving the history
history = model.fit(x_train, y_train_encoded, epochs=10, validation_data=(x_test, y_test_encoded))

# Plotting Training vs. Test Loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Training vs. Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Predicting labels
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Accuracy: {accuracy*100}%')

# Generating a classification report
report = classification_report(y_test, y_pred_classes)
print(report)

# Computing the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred_classes)

# Plotting the heatmap for the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Define different complexities (number of neurons)
complexities = [10, 50, 100, 200]
test_accuracies = []

for complexity in complexities:
    # Create model with different complexities
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(complexity, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train_encoded, epochs=10, verbose=0)  # Set verbose=0 to suppress output

    # Evaluate model on test data
    _, accuracy = model.evaluate(x_test, y_test_encoded, verbose=0)
    test_accuracies.append(accuracy)

# Plot Classification Accuracy vs. Model Complexity
plt.figure(figsize=(6, 4))
plt.plot(complexities, test_accuracies, marker='o')
plt.title('Classification Accuracy vs Model Complexity')
plt.xlabel('Model Complexity (Number of Neurons)')
plt.ylabel('Test Accuracy')
plt.show()
