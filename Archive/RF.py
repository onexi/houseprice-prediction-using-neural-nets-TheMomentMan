import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load and Prepare the Data
# -----------------------------
data = pd.read_csv('train.csv')

# Define the target variable
target = 'SalePrice'

# Identify categorical columns (assumed to be of object type)
categorical_cols = data.select_dtypes(include=['object']).columns

# One-hot encode categorical variables (drop_first to avoid dummy variable trap)
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Ensure target column exists
if target not in data_encoded.columns:
    raise ValueError(f"Target column '{target}' not found in the data.")

# Separate features and target
X = data_encoded.drop(columns=[target])
y = data_encoded[target]

# Ensure all features and target are numeric
X = X.astype(float)
y = y.astype(float)

# -----------------------------
# 2. Split the Data into Training and Test Sets
# -----------------------------
# Use 80% of the data for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Cross-Validation on the Training Set
# -----------------------------
# Set up 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the RandomForestRegressor (using a fixed random state for reproducibility)
model = RandomForestRegressor(random_state=42)

# Compute cross-validated R² scores (higher is better)
r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
# Compute cross-validated MSE (scoring returns negative values for MSE, so we negate them)
mse_scores = -cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')

print("Cross-validated R² scores (training):", r2_scores)
print("Mean cross-validated R² (training):", np.mean(r2_scores))
print("Cross-validated MSE scores (training):", mse_scores)
print("Mean cross-validated MSE (training):", np.mean(mse_scores))

# -----------------------------
# 4. Train the Final Model on the Training Set
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate the Model on the Test Set
# -----------------------------
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nFinal Model Performance on Test Set:")
print("Test MSE:", test_mse)
print("Test R²:", test_r2)
