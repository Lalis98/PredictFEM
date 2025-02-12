import os
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# =================================================== IMPORT DATA ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(BASE_DIR, "../data/test_data.csv")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "../data/predicted_displacements.csv")


df_test = pl.read_csv(TEST_DATA_PATH)[["U_X", "U_Y"]].to_numpy() # Load Actual Test Data
df_pred = pl.read_csv(PREDICTIONS_PATH)[["U_X_pred", "U_Y_pred"]].to_numpy() # Load Model Predictions

# ================================================ METRICS EVALUATION ==================================================

# Mean Absolute Error
mae_x = mean_absolute_error(df_test[:, 0], df_pred[:, 0])
mae_y = mean_absolute_error(df_test[:, 1], df_pred[:, 1])

# Mean Square Error
mse_x = mean_squared_error(df_test[:, 0], df_pred[:, 0])
mse_y = mean_squared_error(df_test[:, 1], df_pred[:, 1])

# Coefficient of Determination R2
r2_x = r2_score(df_test[:, 0], df_pred[:, 0])
r2_y = r2_score(df_test[:, 1], df_pred[:, 1])

print("Model Performance Metrics")
print(f"MAE (U_X): {mae_x:.6f}, MAE (U_Y): {mae_y:.6f}")
print(f"MSE (U_X): {mse_x:.6f}, MSE (U_Y): {mse_y:.6f}")
print(f"R² Score (U_X): {r2_x:.4f}, R² Score (U_Y): {r2_y:.4f}")

# =============================================== PLOT RESULTS =========================================================

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# U_X Plot
axs[0].scatter(df_test[:, 0], df_pred[:, 0], alpha=0.5, color="blue")
axs[0].plot([-0.01, 0.01], [-0.01, 0.01], linestyle="--", color="black")  # Perfect Fit Line
axs[0].set_xlabel("Actual U_X")
axs[0].set_ylabel("Predicted U_X")
axs[0].set_title(f"Predicted vs Actual U_X (R²: {r2_x:.4f})")
axs[0].grid()

# U_Y Plot
axs[1].scatter(df_test[:, 1], df_pred[:, 1], alpha=0.5, color="red")
axs[1].plot([-0.02, 0.02], [-0.02, 0.02], linestyle="--", color="black")  # Perfect Fit Line
axs[1].set_xlabel("Actual U_Y")
axs[1].set_ylabel("Predicted U_Y")
axs[1].set_title(f"Predicted vs Actual U_Y (R²: {r2_y:.4f})")

plt.tight_layout()
axs[1].grid()
plt.show()
