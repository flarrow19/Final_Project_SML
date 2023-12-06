# Importing required libraries
import pandas as pd
import numpy as np
from keras.datasets import fashion_mnist
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Loading the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocessing the data by flattening the images and scaling pixel values
X_train_scaled = x_train.reshape(x_train.shape[0], -1) / 255.0
X_test_scaled = x_test.reshape(x_test.shape[0], -1) / 255.0

# Training SVM models with different kernels on the scaled training data
svm_model_linear = SVC(kernel='linear')
svm_model_linear.fit(X_train_scaled, y_train)

svm_model_radial = SVC(kernel='rbf', gamma='scale')
svm_model_radial.fit(X_train_scaled, y_train)

svm_model_sigmoid = SVC(kernel='sigmoid', gamma='scale', coef0=0)
svm_model_sigmoid.fit(X_train_scaled, y_train)

svm_model_poly = SVC(kernel='poly', degree=3, coef0=1)
svm_model_poly.fit(X_train_scaled, y_train)

# Making predictions with each model on the test set
y_pred_linear = svm_model_linear.predict(X_test_scaled)
y_pred_radial = svm_model_radial.predict(X_test_scaled)
y_pred_sigmoid = svm_model_sigmoid.predict(X_test_scaled)
y_pred_poly = svm_model_poly.predict(X_test_scaled)

# Printing accuracy scores for each model
print("Linear Kernel Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Radial Kernel Accuracy:", accuracy_score(y_test, y_pred_radial))
print("Sigmoid Kernel Accuracy:", accuracy_score(y_test, y_pred_sigmoid))
print("Polynomial Kernel Accuracy:", accuracy_score(y_test, y_pred_poly))

# Generating classification reports for each model
result_linear = classification_report(y_test, y_pred_linear, output_dict=True)
result_radial = classification_report(y_test, y_pred_radial, output_dict=True)
result_sigmoid = classification_report(y_test, y_pred_sigmoid, output_dict=True)
result_poly = classification_report(y_test, y_pred_poly, output_dict=True)

# Converting the classification reports into DataFrame format
df_result_linear = pd.DataFrame(result_linear).transpose()
df_result_radial = pd.DataFrame(result_radial).transpose()
df_result_sigmoid = pd.DataFrame(result_sigmoid).transpose()
df_result_poly = pd.DataFrame(result_poly).transpose()

# Defining the space for hyperparameter optimization
space = {
    'C': hp.loguniform('C', np.log(0.001), np.log(100)),  # Regularization parameter
    'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly'])  # Kernel type
}

# Objective function for hyperparameter tuning
def objective(params):
    model = SVC(C=params['C'], kernel=params['kernel'], gamma='scale')
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    accuracy = scores.mean()
    return {'loss': -accuracy, 'status': STATUS_OK, 'accuracy': accuracy}

# Running the hyperparameter optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,  # Number of evaluations
    trials=trials,
    callbacks=[tqdm(total=50, desc="Hyperopt")]
)

# Retrieving the best accuracy from the trials
best_accuracy = max(trial['result']['accuracy'] for trial in trials.trials)
print("Best hyperparameters:", best)
print("Best accuracy:", best_accuracy)

# Visualizing the results of hyperparameter tuning using a heatmap
# Extracting the scores for each parameter combination
scores = [-trial['result']['loss'] for trial in trials.trials]

# Assuming that we have tried 3 kernels, we reshape the array accordingly
scores_array = np.array(scores).reshape(-1, 3)

# Creating a DataFrame from the scores array
df_scores = pd.DataFrame(scores_array, columns=['linear', 'rbf', 'poly'])

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_scores, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title('Heatmap of SVM Hyperparameter Tuning Results')
plt.xlabel('Kernel Type')
plt.ylabel('Hyperparameter Combinations')
plt.show()

# Create lists to store model complexities and accuracies
complexities = []
train_accuracies = []
test_accuracies = []

# Assuming you have a range of C values to iterate over
C_values = np.logspace(-3, 2, num=10)  # 10 values from 0.001 to 100
for C in tqdm(C_values, desc="Training models"):
    # Train model
    model = SVC(kernel='kernel_of_choice', C=C)
    model.fit(X_train_scaled, y_train)

    # Calculate training and test accuracy
    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

    # Append to lists
    complexities.append(C)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plotting the accuracies
plt.figure(figsize=(12, 6))

# Classification Accuracy vs. Model Complexity
plt.subplot(1, 2, 1)
plt.plot(complexities, test_accuracies, label='Test Accuracy')
plt.plot(complexities, train_accuracies, label='Train Accuracy')
plt.xscale('log')
plt.xlabel('Model Complexity (C value)')
plt.ylabel('Classification Accuracy')
plt.title('Classification Accuracy vs. Model Complexity')
plt.legend()

# Training vs Test Accuracy
plt.subplot(1, 2, 2)
plt.plot(complexities, np.subtract(train_accuracies, test_accuracies), label='Train - Test Accuracy')
plt.xscale('log')
plt.xlabel('Model Complexity (C value)')
plt.ylabel('Difference in Accuracy')
plt.title('Training vs Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
