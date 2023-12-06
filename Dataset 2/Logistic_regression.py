"""
@author: Arpit
"""

# Importing necessary libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import fetch_openml
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Fetching the dataset
data = fetch_openml(name='adult', version=1)
X = data.data
y = data.target

# Convert X to DataFrame if not already
if not isinstance(X, pd.DataFrame):
    X = pd.DataFrame(X)

# Display feature and target names
print(data.feature_names)
print(data.target_names)

# Identifying numerical and categorical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Preprocessing the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])
X_scaled = preprocessor.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred) * 100
print(f'Accuracy: {accuracy}%')
df_result = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
print(df_result)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'max_iter': [500, 1000, 1500],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_scaled, y)
print("Best parameters found: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the heatmap for the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='viridis')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Calculate Training and Test Loss for Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
train_probs = model.predict_proba(X_train)
test_probs = model.predict_proba(X_test)
train_loss = log_loss(y_train, train_probs)
test_loss = log_loss(y_test, test_probs)

# Plotting Training vs. Test Loss
plt.figure(figsize=(6, 4))
plt.bar(['Train Loss', 'Test Loss'], [train_loss, test_loss], color=['blue', 'orange'])
plt.ylabel('Log Loss')
plt.title('Training vs. Test Loss')
plt.show()

# Extracting results from GridSearchCV for Classification Metric vs. Model Complexity
results = pd.DataFrame(grid_search.cv_results_)

mean_scores = results.groupby('param_C')['mean_test_score'].mean()

complexity_values = results['param_C'].unique()
mean_scores.index = pd.to_numeric(mean_scores.index)

# Plotting Classification Metric vs. Model Complexity
plt.figure(figsize=(10, 6))
plt.plot(mean_scores.index, mean_scores.values, marker='o')
plt.xscale('log')
plt.xlabel('C (Model Complexity)')
plt.ylabel('Mean Test Accuracy')
plt.title('Classification Accuracy vs. Model Complexity')
plt.show()
