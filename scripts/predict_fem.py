import os
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from scripts.generate_fem_data import NUM_X, NUM_Y
from scripts.train_ai_model import NUM_FEATURES, OUTPUT_DIM

# ========================================== IMPORT DATA ===============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(BASE_DIR, "../data/processed_test_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/fem_lstm.pth")
OUTPUT_PATH = os.path.join(BASE_DIR, "../data/predicted_displacements.csv")

if not os.path.exists(MODEL_PATH): # Ensure Model File Exists
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")


df_test = pl.read_csv(TEST_DATA_PATH) # Load Processed Test Data (Already Normalized)

# ============================================ CONSTANTS ===============================================================

NUM_NODES = NUM_X * NUM_Y

# =========================================== RESHAPE DATA =============================================================

# Reshape Test Data to (batch_size, 100, num_features)
num_analyses = len(df_test) // NUM_NODES
X_test = df_test[['X', 'Y', 'E', 'POISSON_RATIO', 'THICKNESS', 'FIXED_X', 'FIXED_Y', 'FORCE_Y']].to_numpy()
X_test_reshaped = X_test.reshape(num_analyses, NUM_NODES, NUM_FEATURES)

# ========================================= PYTORCH TENSORS DATA TYPE ==================================================
X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)

# ============================================== LSTM CLASS MODEL ======================================================
class FEMLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(FEMLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Fully Connected for U_X & U_Y

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Output shape: (batch_size, 100, hidden_dim)
        output = self.fc(lstm_out)  # Shape: (batch_size, 100, 2)
        return output

# ============================================== LSTM CLASS MODEL ======================================================

HIDDEN_DIM = 128
NUM_LAYERS = 3
model = FEMLSTM(input_dim=NUM_FEATURES, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS)

# Load Model Weights (Ensure Correct Device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# ============================================== RUN PREDICTIONS =======================================================

with torch.no_grad():
    Y_pred_tensor = model(X_test_tensor)

Y_pred = Y_pred_tensor.numpy().reshape(-1, OUTPUT_DIM)   # Convert Predictions to NumPy Array

# =========================================== DENORMALIZE PREDICTIONS ==================================================

SCALER_PATH = os.path.join(BASE_DIR, "../data/scaler_metadata.npz")

if os.path.exists(SCALER_PATH):
    scaler_data = np.load(SCALER_PATH)  # Load Min-Max scaling parameters

    # Extract min/max values for displacement predictions
    U_X_min, U_X_max = scaler_data["U_X_min"], scaler_data["U_X_max"]
    U_Y_min, U_Y_max = scaler_data["U_Y_min"], scaler_data["U_Y_max"]

    # Apply inverse scaling
    Y_pred[:, 0] = Y_pred[:, 0] * (U_X_max - U_X_min) + U_X_min  # U_X
    Y_pred[:, 1] = Y_pred[:, 1] * (U_Y_max - U_Y_min) + U_Y_min  # U_Y

# =============================================== SAVE PREDICTIONS =====================================================

df_pred = pl.DataFrame({"U_X_pred": Y_pred[:, 0], "U_Y_pred": Y_pred[:, 1]})
df_pred.write_csv(OUTPUT_PATH)
print(f"Predictions saved to {OUTPUT_PATH}")
