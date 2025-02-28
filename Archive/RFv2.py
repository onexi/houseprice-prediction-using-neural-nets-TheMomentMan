import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load and Prepare the Data
# -----------------------------
data = pd.read_csv('train.csv')
print(data.shape)
target = 'SalePrice'

# Identify categorical columns (assumed to be of object type)
categorical_cols = data.select_dtypes(include=['object']).columns

# One-hot encode categorical variables (drop_first to avoid dummy variable trap)
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Replace any infinite values with NaN and drop rows with missing data
data_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
data_encoded.dropna(inplace=True)

print(data_encoded.shape)

# Separate features and target
X = data_encoded.drop(columns=[target])
y = data_encoded[target]

# Ensure all features and target are numeric
X = X.astype(float)
y = y.astype(float)

# -----------------------------
# 2. Split into Training and Test Sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 3. Hyperparameter Tuning using GridSearchCV
# -----------------------------
# Define a hyperparameter grid. You can adjust the values as needed.
param_grid = {
    'n_estimators': [100, 200, 300],         # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],           # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],           # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],             # Minimum samples required at a leaf node
    'bootstrap': [True, False]                 # Whether bootstrap samples are used
}

# Initialize the RandomForestRegressor with a fixed random state for reproducibility.
rf = RandomForestRegressor(random_state=42)

# Set up GridSearchCV. We use negative MSE because GridSearchCV expects a score to maximize.
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, scoring='neg_mean_squared_error',
                           n_jobs=-1, verbose=1)

# Fit GridSearchCV on the training data.
grid_search.fit(X_train, y_train)

print("Best hyperparameters found:", grid_search.best_params_)
print("Best cross-validated score (negative MSE):", grid_search.best_score_)

# -----------------------------
# 4. Evaluate the Best Model on the Test Set
# -----------------------------
best_rf = grid_search.best_estimator_
y_test_pred = best_rf.predict(X_test)

test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nFinal Model Performance on Test Set:")
print("Test MSE:", test_mse)
print("Test R²:", test_r2)
