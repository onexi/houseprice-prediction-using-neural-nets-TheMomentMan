# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import itertools

# -----------------------------
# 1. Load and Prepare the Data
# -----------------------------
data = pd.read_csv('train.csv')

# Define target
target = 'SalePrice'

# Identify categorical columns (assumed object type) and one-hot encode
categorical_cols = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Drop rows with missing values (if any)
data_encoded.dropna(inplace=True)

# Separate features and target
X = data_encoded.drop(columns=[target])
y = data_encoded[target]

# Ensure numeric type
X = X.astype(float)
y = y.astype(float)

# -----------------------------
# 2. Split Data: Train, Validation, and Test
# -----------------------------
# First, split train+val and test (80/20)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Then split train and validation (e.g., 75/25 of trainval => 60/20 overall)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42
)

# -----------------------------
# 3. Standardize Features
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
X_trainval = scaler.fit_transform(X_trainval)  # For retraining later, you might fit on trainval

# -----------------------------
# 4. Convert Data to PyTorch Tensors
# -----------------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)

# -----------------------------
# 5. Define the Neural Network Model
# -----------------------------
class HousePriceNN(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout=0.0):
        super(HousePriceNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x):
        x = self.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.relu(self.fc2(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.fc3(x)
        return x

# -----------------------------
# 6. Define a Function to Train and Evaluate the Model
# -----------------------------
def train_and_evaluate(hparams, device='cpu'):
    """
    hparams: dict with keys:
      - learning_rate
      - batch_size
      - num_epochs
      - hidden1
      - hidden2
      - dropout
    Trains on the training set and returns the validation MSE.
    """
    input_dim = X_train_tensor.shape[1]
    model = HousePriceNN(input_dim, hidden1=hparams['hidden1'],
                         hidden2=hparams['hidden2'],
                         dropout=hparams['dropout']).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    
    for epoch in range(hparams['num_epochs']):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor.to(device))
        val_loss = criterion(val_preds, y_val_tensor.to(device)).item()
    return val_loss, model

# -----------------------------
# 7. Hyperparameter Tuning via Grid Search
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a hyperparameter grid dictionary.
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32],
    'num_epochs': [100, 200],
    'hidden1': [64, 128],
    'hidden2': [32, 64],
    'dropout': [0.0, 0.2]
}

best_val_mse = float('inf')
best_params = None
best_model_state = None

# Iterate over all combinations
for lr, bs, ep, h1, h2, dp in itertools.product(
    param_grid['learning_rate'],
    param_grid['batch_size'],
    param_grid['num_epochs'],
    param_grid['hidden1'],
    param_grid['hidden2'],
    param_grid['dropout']
):
    hparams = {
        'learning_rate': lr,
        'batch_size': bs,
        'num_epochs': ep,
        'hidden1': h1,
        'hidden2': h2,
        'dropout': dp
    }
    val_mse, temp_model = train_and_evaluate(hparams, device=device)
    print(f"Params: {hparams} -> Validation MSE: {val_mse:.2f}")
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_params = hparams
        best_model_state = temp_model.state_dict()

print("\nBest hyperparameters found:")
print(best_params)
print(f"Best validation MSE: {best_val_mse:.2f}")

# -----------------------------
# 8. Retrain Final Model on Training + Validation Data
# -----------------------------
# Combine training and validation sets.
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train.values.reshape(-1, 1), y_val.values.reshape(-1, 1)], axis=0)
X_trainval_tensor = torch.tensor(X_trainval, dtype=torch.float32)
y_trainval_tensor = torch.tensor(y_trainval, dtype=torch.float32).view(-1, 1)
trainval_dataset = TensorDataset(X_trainval_tensor, y_trainval_tensor)

def train_final_model(hparams, device='cpu'):
    input_dim = X_trainval_tensor.shape[1]
    model = HousePriceNN(input_dim, hidden1=hparams['hidden1'],
                         hidden2=hparams['hidden2'],
                         dropout=hparams['dropout']).to(device)
    train_loader = DataLoader(trainval_dataset, batch_size=hparams['batch_size'], shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    
    for epoch in range(hparams['num_epochs']):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

final_model = train_final_model(best_params, device=device)

# -----------------------------
# 9. Evaluate Final Model on the Test Set
# -----------------------------
final_model.eval()
with torch.no_grad():
    test_preds = final_model(X_test_tensor.to(device))
    final_test_mse = mean_squared_error(y_test, test_preds.cpu().numpy())
    final_test_r2  = r2_score(y_test, test_preds.cpu().numpy())

print("\nFinal Model Performance on Test Set:")
print("Test MSE:", final_test_mse)
print("Test R²:", final_test_r2)