# **PredictFEM: FEM-Based Displacement Prediction Using LSTM**

## **📌 Overview**
PredictFEM is a framework that combines **Finite Element Method (FEM) simulations** with **machine learning** to predict displacements in a **2D linear isotropic elastic plate under plane stress conditions**. The system first solves FEM problems and then trains a **Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN)** to predict displacement fields based on FEM-generated data.

---

## **📁 Project Structure**

```
PredictFEM/
│── main.py                   # Entry point for running predictions
│── scripts/                   # Contains scripts for data generation, preprocessing, training, and evaluation
│   ├── preprocess_data.py      # Normalizes FEM data before training
│   ├── generate_fem_data.py    # Generates FEM-based displacement data
│   ├── train_ai_model.py       # Trains the LSTM model on FEM data
│   ├── predict_fem.py          # Runs the trained LSTM model to predict displacements
│   ├── evaluate_predictions.py # Computes error metrics for model evaluation
│── fem_solver/                # Custom FEM solver implementation
│   ├── __init__.py             # Initializes the FEM solver package
│   ├── dof.py                  # Defines degrees of freedom for FEM nodes
│   ├── element.py              # Represents FEM elements and stiffness matrices
│   ├── material.py             # Defines material properties (e.g., Young’s modulus, Poisson’s ratio)
│   ├── mesh.py                 # Handles FEM mesh generation and node connectivity
│   ├── node.py                 # Defines individual FEM nodes
│   ├── solver.py               # Solves FEM equations for displacement computation
│   ├── utils.py                # Utility functions for FEM operations
│── models/                     # Stores trained LSTM model files
│   ├── fem_lstm.pth            # Pre-trained LSTM model
│── data/                       # Contains input/output FEM data
│   ├── train_data.csv          # Training dataset
│   ├── test_data.csv           # Testing dataset
│   ├── predicted_displacements.csv # LSTM model predictions
│── requirements.txt             # List of dependencies for running the project
│── .gitignore                   # Specifies files and folders to ignore in version control
│── README.md                    # Project documentation
```

---

## **🚀 Installation**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/yourusername/PredictFEM.git
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Generate FEM Data**
Run the script to create a dataset using the FEM solver:

```sh
python scripts/generate_fem_data.py
```

### **4️⃣ Train the LSTM Model**
Train the LSTM model on the generated FEM data:

```sh
python scripts/train_ai_model.py
```


### **5️⃣ Run Predictions**
Use the trained model to predict displacements:

```sh
python scripts/predict_fem.py
```

### **6️⃣ Evaluate Model Performance**
Compute error metrics and visualize predictions:

```sh
python scripts/evaluate_predictions.py
```

## **📌 Features**
 - 🏗 Custom FEM Solver: Computes displacements using an in-house finite element analysis engine.
 - 🤖 LSTM-Based Prediction: Learns from FEM data to approximate displacement fields.
 - 📊 Performance Evaluation: Uses Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score to assess model accuracy.
 - 🔄 Data Preprocessing: Normalizes input data and applies MinMax scaling.
 - 💾 Reproducible Workflow: Well-structured pipeline from data generation to prediction.

## **🛠 Technologies Used**
 - Python 3.8+
 - NumPy & Polars (Data handling)
 - PyTorch (LSTM model)
 - Scikit-Learn (Data normalization & evaluation metrics)
 - Matplotlib (Visualization)

## **📊 Model Performance**
Example results comparing FEM vs. LSTM predictions:

| Metric    | U_X (X-Displacement) | U_Y (Y-Displacement) |
|-----------|----------------------|----------------------|
| **MAE**   | 0.0023               | 0.0019               |
| **MSE**   | 0.00005              | 0.00004              |
| **R² Score** | 0.98             | 0.97                 |

## **📄 License**
This project is licensed under the MIT License.



## **👨‍💻 Contributors**
[Michalis Lefkiou] - Naval Architect & Marine Engineer | Computational Mechanics
[Other Contributors]

## **📬 Contact**
For questions or collaboration: 📧 Email: michalis.leukioug1@gmail.com
🔗 GitHub: [yourgithubprofile](https://github.com/Lalis98)

