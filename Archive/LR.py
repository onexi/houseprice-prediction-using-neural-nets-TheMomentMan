import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load the Dataset
# -----------------------------
data = pd.read_csv('train.csv')

# Define the target variable
target = 'SalePrice'

# -----------------------------
# 2. One-Hot Encode Categorical Columns
# -----------------------------
categorical_cols = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Replace any infinite values with NaN and drop rows with missing data
data_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
data_encoded.dropna(inplace=True)

# -----------------------------
# 3. Separate Features and Target
# -----------------------------
X = data_encoded.drop(columns=[target])
y = data_encoded[target]

# Convert all columns to numeric type
X = X.astype(float)
y = y.astype(float)

# Add constant to the entire DataFrame (so that both train and val have it)
X = sm.add_constant(X)

# -----------------------------
# 4. Split the Data into Training and Validation Sets
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 5. Fit the Linear Regression Model on Training Data
# -----------------------------
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

# -----------------------------
# 6. Evaluate the Model on the Validation Set
# -----------------------------
y_val_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print("\nValidation Mean Squared Error (MSE):", mse)
print("Validation R²:", r2)
