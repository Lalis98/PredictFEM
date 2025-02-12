import os
import numpy as np
import polars as pl
from fem_solver import Mesh, Material, Solver

NUM_X, NUM_Y = 10, 10

def generate_fem_data(num_train=100, num_test=20):
    """
    Generates FEM simulation data, splitting into training and test sets.

    :param num_train: Number of training samples (default: 100).
    :param num_test: Number of test samples (default: 20).
    """

    # ======================================= INITIALIZE DATA SETS =====================================================
    data_train, data_test = [], []

    for i in range(num_train + num_test): # Run the number of analysis times

        # ========================================== VARIABLES =========================================================
        E = np.random.uniform(100E9, 300E9)
        poisson_ratio = np.random.uniform(0.25, 0.35)
        force_value = np.random.uniform(-5E5, -20E5)

        LENGTH, HEIGHT, THICKNESS = 1.0, 1.0, 0.1

        # ========================================= FEM ANALYSIS =======================================================
        mesh = Mesh()
        mesh.create_rectangular_mesh(NUM_X, NUM_Y, LENGTH, HEIGHT, THICKNESS)
        material = Material("Steel", E, poisson_ratio, "Plane Stress")
        solver = Solver(mesh, material)
        solver.assemble_global_stiffness()

        # ======================================= BOUNDARY CONDITIONS ==================================================

        boundary_conditions = {node.node_id: ("ALL", 0.0) for node in mesh.nodes if node.coordinates['X'] == 0.0}
        solver.apply_boundary_conditions(boundary_conditions)

        # ============================================== LOADS =========================================================

        force_nodes = [node for node in mesh.nodes if
                       node.coordinates['X'] == max(node.coordinates['X'] for node in mesh.nodes)]
        for node in force_nodes:
            solver.apply_force(node_id=node.node_id, direction="Y", force_value=force_value)

        # =========================================== SOLVE PROBLEM ====================================================

        U = solver.solve()

        # ============================================= STORE DATA =====================================================

        dataset = data_train if i < num_train else data_test  # Decide which dataset to store it in
        for node in mesh.nodes:
            x, y = node.coordinates['X'], node.coordinates['Y']
            fixed_x = 1 if node.node_id in boundary_conditions else 0
            fixed_y = 1 if node.node_id in boundary_conditions else 0
            dof_indices = node.get_dof_indices()
            u_x, u_y = U[dof_indices['X']], U[dof_indices['Y']]

            dataset.append(
                [x, y, E, poisson_ratio, THICKNESS, fixed_x, fixed_y, node.dof.force["X"], node.dof.force["Y"], u_x,
                 u_y])

        print(f"Simulation {i + 1}/{num_train + num_test} completed!")

    # ============================================== SAVE DATA =========================================================

    columns = ["X", "Y", "E", "POISSON_RATIO", "THICKNESS", "FIXED_X", "FIXED_Y", "FORCE_X", "FORCE_Y", "U_X", "U_Y"]
    df_train = pl.DataFrame(data_train, schema=columns, orient="row")
    df_test = pl.DataFrame(data_test, schema=columns, orient="row")

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    os.makedirs(DATA_DIR, exist_ok=True)

    df_train.write_csv(os.path.join(DATA_DIR, "train_data.csv"))
    df_test.write_csv(os.path.join(DATA_DIR, "test_data.csv"))

    print("Training Data saved as `data/train_data.csv`")
    print("Test Data saved as `data/test_data.csv`")


if __name__ == "__main__":
    generate_fem_data(num_train=10000, num_test=2000)
