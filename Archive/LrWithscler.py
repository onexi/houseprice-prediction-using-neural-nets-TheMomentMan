import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load and Prepare the Data
# -----------------------------
data = pd.read_csv('train.csv')
target = 'SalePrice'

# Identify categorical columns (assumed to be of object type)
categorical_cols = data.select_dtypes(include=['object']).columns

# One-hot encode categorical variables (drop_first to avoid dummy variable trap)
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Drop rows with missing values
data_encoded.dropna(inplace=True)

# Separate features and target
X = data_encoded.drop(columns=[target])
y = data_encoded[target]

# Ensure numeric type
X = X.astype(float)
y = y.astype(float)

# -----------------------------
# 2. Split the Data into Training and Test Sets
# -----------------------------
# Use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Standardize the Features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Force the addition of a constant column for both train and test sets
X_train_scaled = sm.add_constant(X_train_scaled, has_constant='add')
X_test_scaled  = sm.add_constant(X_test_scaled, has_constant='add')

# -----------------------------
# 4. Fit the Linear Regression Model
# -----------------------------
model = sm.OLS(y_train, X_train_scaled).fit()
print(model.summary())

# -----------------------------
# 5. Evaluate the Model on the Test Set
# -----------------------------
y_test_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\nTest Mean Squared Error (MSE):", mse)
print("Test R²:", r2)
