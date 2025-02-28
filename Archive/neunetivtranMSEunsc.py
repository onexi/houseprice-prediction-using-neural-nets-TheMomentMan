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
target = 'SalePrice'

# Identify categorical columns (assumed object type) and one-hot encode them
categorical_cols = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Drop rows with missing values
data_encoded.dropna(inplace=True)

# Separate features and target; ensure numeric types
X = data_encoded.drop(columns=[target]).astype(float)
y = data_encoded[target].astype(float)

# -----------------------------
# 2. Split Data: Train, Validation, and Test
# -----------------------------
# Split into train+val and test (80/20)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Then split train+val into train and validation (approx 60% train, 20% val overall)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42
)

# -----------------------------
# 3. Standardize Features and Scale the Target (y)
# -----------------------------
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val   = scaler_X.transform(X_val)
X_test  = scaler_X.transform(X_test)
# (For final retraining, we may use X_trainval as well)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1))
y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1, 1))

# -----------------------------
# 4. Convert Data to PyTorch Tensors
# -----------------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val_scaled, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test_scaled, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)

# -----------------------------
# 5. Define a Deeper Neural Network Model with Dropout
# -----------------------------
class HousePriceNNDeep(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, hidden3=32, dropout=0.5):
        super(HousePriceNNDeep, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 1)  # Regression output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# -----------------------------
# 6. Define Training Function with Early Stopping and Weight Decay
# -----------------------------
def train_and_evaluate_deep(hparams, device='cpu', patience=20):
    """
    Trains the deep network using provided hyperparameters.
    Incorporates weight decay and early stopping based on validation loss.
    Returns:
       - best_val_loss (scaled MSE) [for early stopping],
       - model (final best model),
       - unscaled_val_mse,
       - unscaled_val_r2.
       
    hparams: dict with keys:
      - learning_rate, batch_size, num_epochs,
      - hidden1, hidden2, hidden3, dropout
    """
    input_dim = X_train_tensor.shape[1]
    model = HousePriceNNDeep(input_dim, hidden1=hparams['hidden1'],
                             hidden2=hparams['hidden2'],
                             hidden3=hparams['hidden3'],
                             dropout=hparams['dropout']).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'], weight_decay=1e-5)
    
    best_val_loss = float('inf')
    epochs_without_improve = 0
    best_model_state = None
    
    for epoch in range(hparams['num_epochs']):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        epoch_loss = running_loss / len(train_dataset)
        
        # Evaluate on validation set (scaled values)
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor.to(device))
            val_loss = criterion(val_preds, y_val_tensor.to(device)).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improve += 1
        
        if epochs_without_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}, best validation loss (scaled): {best_val_loss:.4f}")
            break

    # Load best model
    model.load_state_dict(best_model_state)
    
    # Compute unscaled performance on the validation set:
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor.to(device))
        # Convert predictions from tensor to numpy array
        val_preds_np = val_preds.cpu().numpy()
        y_val_np = y_val_tensor.cpu().numpy()
        
        # Inverse transform predictions and true values
        val_preds_unscaled = scaler_y.inverse_transform(val_preds_np)
        y_val_unscaled = scaler_y.inverse_transform(y_val_np)
        
        unscaled_val_mse = mean_squared_error(y_val_unscaled, val_preds_unscaled)
        unscaled_val_r2 = r2_score(y_val_unscaled, val_preds_unscaled)
    
    return best_val_loss, model, unscaled_val_mse, unscaled_val_r2

# -----------------------------
# 7. Hyperparameter Tuning via Grid Search for the Deep Model
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

param_grid = {
    'learning_rate': [0.001, 0.01],
    'batch_size': [16, 32],
    'num_epochs': [100, 200],
    'hidden1': [128, 256],
    'hidden2': [64, 128],
    'hidden3': [32, 64],
    'dropout': [0.2, 0.5]
}

best_val_mse = float('inf')
best_params = None
best_model_state = None
best_unscaled_val_mse = None
best_unscaled_val_r2 = None

for lr, bs, ep, h1, h2, h3, dp in itertools.product(
    param_grid['learning_rate'],
    param_grid['batch_size'],
    param_grid['num_epochs'],
    param_grid['hidden1'],
    param_grid['hidden2'],
    param_grid['hidden3'],
    param_grid['dropout']
):
    hparams = {
        'learning_rate': lr,
        'batch_size': bs,
        'num_epochs': ep,
        'hidden1': h1,
        'hidden2': h2,
        'hidden3': h3,
        'dropout': dp
    }
    val_loss_scaled, temp_model, val_mse_unscaled, val_r2_unscaled = train_and_evaluate_deep(hparams, device=device, patience=20)
    print(f"Params: {hparams} -> Scaled Val Loss: {val_loss_scaled:.4f} | Unscaled Val MSE: {val_mse_unscaled:.2f}, R²: {val_r2_unscaled:.4f}")
    if val_loss_scaled < best_val_mse:
        best_val_mse = val_loss_scaled
        best_params = hparams
        best_model_state = temp_model.state_dict()
        best_unscaled_val_mse = val_mse_unscaled
        best_unscaled_val_r2 = val_r2_unscaled

print("\nBest hyperparameters found for the deep network:")
print(best_params)
print(f"Best validation loss (scaled): {best_val_mse:.4f}")
print(f"Unscaled Validation MSE: {best_unscaled_val_mse:.2f}")
print(f"Unscaled Validation R²: {best_unscaled_val_r2:.4f}")

# -----------------------------
# 8. Retrain Final Model on Training + Validation Data
# -----------------------------
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train_scaled, y_val_scaled], axis=0)
X_trainval_tensor = torch.tensor(X_trainval, dtype=torch.float32)
y_trainval_tensor = torch.tensor(y_trainval, dtype=torch.float32)
trainval_dataset = TensorDataset(X_trainval_tensor, y_trainval_tensor)

def train_final_deep_model(hparams, device='cpu'):
    input_dim = X_trainval_tensor.shape[1]
    model = HousePriceNNDeep(input_dim, hidden1=hparams['hidden1'],
                             hidden2=hparams['hidden2'],
                             hidden3=hparams['hidden3'],
                             dropout=hparams['dropout']).to(device)
    train_loader = DataLoader(trainval_dataset, batch_size=hparams['batch_size'], shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'], weight_decay=1e-5)
    
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

final_deep_model = train_final_deep_model(best_params, device=device)

# -----------------------------
# 9. Evaluate Final Deep Model on the Test Set (Inverse-transform to Original Scale)
# -----------------------------
final_deep_model.eval()
with torch.no_grad():
    test_preds = final_deep_model(X_test_tensor.to(device))
    test_preds_np = test_preds.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()
    test_preds_unscaled = scaler_y.inverse_transform(test_preds_np)
    y_test_unscaled = scaler_y.inverse_transform(y_test_np)
    final_test_mse = mean_squared_error(y_test_unscaled, test_preds_unscaled)
    final_test_r2  = r2_score(y_test_unscaled, test_preds_unscaled)

print("\nFinal Deep Model Performance on Test Set:")
print("Test MSE:", final_test_mse)
print("Test R²:", final_test_r2)
