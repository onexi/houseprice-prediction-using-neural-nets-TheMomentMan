import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from itertools import product

# -----------------------------
# 1. Load and Prepare Data
# -----------------------------
data = pd.read_csv('train.csv')
numeric_cols = data.select_dtypes(include=[np.number]).columns
clean_numeric_cols = [col for col in numeric_cols if data[col].isna().sum() == 0]
data_clean = data[clean_numeric_cols]

if 'SalePrice' not in data_clean.columns:
    raise ValueError("The target column 'SalePrice' is not present in the numeric data.")

corr_matrix = data_clean.corr()
target_corr = corr_matrix['SalePrice'].drop('SalePrice').abs().sort_values(ascending=False)
top4_features = target_corr.head(4).index
print("Selected top 4 features:", list(top4_features))

X = data_clean[top4_features].values
y = data_clean['SalePrice'].values.reshape(-1, 1)

# Split into train/test (and then train/val from train)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor)

# -----------------------------
# 2. Define the Neural Network Model with Dropout
# -----------------------------
class HousePriceModel(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout=0.0):
        super(HousePriceModel, self).__init__()
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
# 3. Training & Evaluation Function
# -----------------------------
def train_and_evaluate(learning_rate, batch_size, num_epochs, hidden1, hidden2, dropout, device='cpu'):
    model = HousePriceModel(
        input_dim=X_train_tensor.shape[1],
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=dropout
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor.to(device))
        val_loss = criterion(val_preds, y_val_tensor.to(device)).item()
    return val_loss, model

# -----------------------------
# 4. Hyperparameter Grid & Search
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the hyperparameter grid using a dictionary.
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'num_epochs': [200, 500, 1000],
    'hidden1': [32, 64, 128],
    'hidden2': [16, 32, 64],
    'dropout': [0.0, 0.2, 0.5]
}

best_val_mse = float('inf')
best_params = {}
best_model_state = None

# Iterate over all combinations using itertools.product.
for lr, bs, ep, h1, h2, dp in product(
        param_grid['learning_rate'],
        param_grid['batch_size'],
        param_grid['num_epochs'],
        param_grid['hidden1'],
        param_grid['hidden2'],
        param_grid['dropout']):
    val_mse, temp_model = train_and_evaluate(lr, bs, ep, h1, h2, dp, device=device)
    print(f"Params: lr={lr}, bs={bs}, ep={ep}, hidden1={h1}, hidden2={h2}, dropout={dp} -> Val MSE: {val_mse:.2f}")
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_params = {
            'learning_rate': lr,
            'batch_size': bs,
            'num_epochs': ep,
            'hidden1': h1,
            'hidden2': h2,
            'dropout': dp
        }
        best_model_state = temp_model.state_dict()

print("\nBest hyperparameters:", best_params)
print(f"Best validation MSE: {best_val_mse:.4f}")

# -----------------------------
# 5. Retrain on (train+val) with Best Hyperparameters
# -----------------------------
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.vstack([y_train, y_val])
X_trainval_tensor = torch.tensor(X_trainval, dtype=torch.float32)
y_trainval_tensor = torch.tensor(y_trainval, dtype=torch.float32)
trainval_dataset = TensorDataset(X_trainval_tensor, y_trainval_tensor)

def train_final_model(lr, bs, ep, h1, h2, dp, device='cpu'):
    model_final = HousePriceModel(
        input_dim=X_trainval_tensor.shape[1],
        hidden1=h1,
        hidden2=h2,
        dropout=dp
    ).to(device)
    trainval_loader = DataLoader(trainval_dataset, batch_size=bs, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_final.parameters(), lr=lr)
    
    for epoch in range(ep):
        model_final.train()
        for batch_X, batch_y in trainval_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model_final(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model_final

final_model = train_final_model(
    best_params['learning_rate'],
    best_params['batch_size'],
    best_params['num_epochs'],
    best_params['hidden1'],
    best_params['hidden2'],
    best_params['dropout'],
    device=device
)

# -----------------------------
# 6. Evaluate on the Test Set
# -----------------------------
final_model.eval()
with torch.no_grad():
    test_preds = final_model(X_test_tensor.to(device))
    test_criterion = nn.MSELoss()
    test_mse = test_criterion(test_preds, y_test_tensor.to(device)).item()

print(f"\nFinal Model Test MSE: {test_mse:.4f}")

# Also evaluate using scikit-learn's MSE:
test_preds_np = test_preds.cpu().numpy()
mse_sklearn = mean_squared_error(y_test, test_preds_np)
print("Test MSE (scikit-learn):", mse_sklearn)

# Show a few sample predictions:
print("\nSample predictions:")
for i in range(5):
    print(f"True: {y_test[i][0]:.2f}, Predicted: {test_preds_np[i][0]:.2f}")
