import os
import polars as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_fem_data(data_type="train"):
    """
    Loads FEM dataset, normalizes numerical values, and saves processed data dynamically
    in the project's `data/` directory.

    :param data_type: "train" or "test" (default: "train").
    """

    # ============================================== IMPORT DATA =======================================================

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get project root directory

    data_dir = os.path.join(project_root, "data")
    input_file = os.path.join(data_dir, f"{data_type}_data.csv")
    output_file = os.path.join(data_dir, f"processed_{data_type}_data.csv")
    scaler_file = os.path.join(data_dir, "scaler_metadata.npz")

    # Load dataset
    print(f"Loading FEM {data_type} dataset from `{input_file}`...")
    df = pl.read_csv(input_file)

    # ============================================ NORMALIZE DATA ======================================================

    feature_columns = ["X", "Y", "E", "POISSON_RATIO", "THICKNESS", "FORCE_X", "FORCE_Y", "U_X", "U_Y"] # Features to normalize

    # Compute min/max before normalization
    min_vals = {col: df[col].min() for col in feature_columns}
    max_vals = {col: df[col].max() for col in feature_columns}

    # Save min/max values (Only when processing training data)
    if data_type == "train":
        np.savez(scaler_file, **{f"{col}_min": min_vals[col] for col in feature_columns},
                 **{f"{col}_max": max_vals[col] for col in feature_columns})
        print(f"Saved Min-Max Scaling Parameters to `{scaler_file}`")

    # Apply Min-Max Scaling (0 to 1)
    scaler = MinMaxScaler()
    df_scaled = df.with_columns([
        pl.Series(col, scaler.fit_transform(df[col].to_numpy().reshape(-1, 1)).flatten())
        for col in feature_columns
    ])

    # ============================================= SAVE DATA ==========================================================

    df_scaled.write_csv(output_file)
    print(f"Preprocessed {data_type} data saved as `{output_file}`")


if __name__ == "__main__":
    preprocess_fem_data("train")  # Run for training data
    preprocess_fem_data("test")   # Run for test data
