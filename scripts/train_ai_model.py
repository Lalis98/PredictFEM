import os
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scripts.generate_fem_data import NUM_X, NUM_Y

# =============================================== LOAD DATA ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/processed_train_data.csv")

df = pl.read_csv(DATA_PATH) # Load Data

# =============================================== CONSTANTS ============================================================

NUM_NODES = NUM_X * NUM_Y  # Nodes per FEM Analysis
NUM_FEATURES = 8  # Input Features: [X, Y, E, Poisson, Thickness, Fixed_X, Fixed_Y, Force_Y]
OUTPUT_DIM = 2    # Predict [U_X, U_Y] for Each Node
SEQ_LENGTH = NUM_NODES  # LSTM Sequences = 100 Nodes Per Analysis from Data

# ============================================== RESHAPE DATA ==========================================================

# Reshape Data to (batch_size, 100, num_features)
num_analyses = len(df) // NUM_NODES  # Number of full FEM cases
X = df[['X', 'Y', 'E', 'POISSON_RATIO', 'THICKNESS', 'FIXED_X', 'FIXED_Y', 'FORCE_Y']].to_numpy()
Y = df[['U_X', 'U_Y']].to_numpy()

# Ensure Data is Grouped by Analysis
X_reshaped = X.reshape(num_analyses, NUM_NODES, NUM_FEATURES)  # Shape: (samples, 100, 8)
Y_reshaped = Y.reshape(num_analyses, NUM_NODES, OUTPUT_DIM)  # Shape: (samples, 100, 2)

# ========================================= PYTORCH TENSORS DATA TYPE ==================================================
X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
Y_tensor = torch.tensor(Y_reshaped, dtype=torch.float32)

# =============================================== BATCH ANALYSIS =======================================================
BATCH_SIZE = 32
dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # Shuffling only at the case level

# ============================================== LSTM CLASS MODEL ======================================================

# Define LSTM-Based Model
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

# Initialize Model, Loss, Optimizer
HIDDEN_DIM = 128
NUM_LAYERS = 3
model = FEMLSTM(input_dim=NUM_FEATURES, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# ================================================== TRAINING ==========================================================

# Training Loop (Batches Contain Full Analyses)
NUM_EPOCHS = 100
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)  # Model predicts displacements
        loss = criterion(predictions, batch_Y)  # MSE Loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.6f}")

# ================================================= SAVE THE MODEL =====================================================

MODEL_PATH = os.path.join(BASE_DIR, "../models/fem_lstm.pth")
torch.save(model.state_dict(), MODEL_PATH)  # Save Model
print(f"Model saved to {MODEL_PATH}")