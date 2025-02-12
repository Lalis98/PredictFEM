import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from fem_solver.utils import (
    shape_function_Q4,
    gauss_quadrature,
    compute_J_matrix,
    compute_B_matrix
)

class Solver:
    """
    Finite Element Solver for Plane Stress/Strain Problems.

    This class performs:
        - Assembly of the global stiffness matrix.
        - Application of boundary conditions and forces.
        - Solving for nodal displacements.
        - Calculation of element stresses.
        - Visualization of deformed structure and stress distribution.
    """

    def __init__(self, mesh, material):
        """
        Initializes the FEM solver.
        """
        self.mesh = mesh
        self.material = material

        self.num_dof = len(mesh.nodes) * 2  # Each node has 2 DOFs (X, Y)
        self.K_global = np.zeros((self.num_dof, self.num_dof))  # Global stiffness matrix
        self.F_global = np.zeros(self.num_dof)  # Global force vector
        self.U = np.zeros(self.num_dof)  # Solution vector (displacements)

    def assemble_global_stiffness(self):
        """
        Assembles the global stiffness matrix from all elements.
        """
        for element in self.mesh.elements:
            ke = self.element_stiffness_matrix(element)

            # Get global DOF indices
            dof_indices = element.get_dof_indices()

            # Assemble into global stiffness matrix
            for i in range(len(dof_indices)):
                for j in range(len(dof_indices)):
                    self.K_global[dof_indices[i], dof_indices[j]] += ke[i, j]

    def element_stiffness_matrix(self, element):
        """
        Computes the stiffness matrix for a quadrilateral element using isoparametric formulation.

        :param element: The element object.
        :type element: Element
        :return: The element stiffness matrix (8x8).
        :rtype: numpy.ndarray
        """
        t = self.mesh.thickness  # Thickness from the mesh
        C = self.material.C  # Constitutive matrix

        # Get nodal coordinates
        xy_coord = np.array(element.get_node_coordinates())
        ke = np.zeros((8, 8))  # 4 nodes × 2 DOF per node

        # Gaussian quadrature points (2x2 integration)
        gauss_points, weights = gauss_quadrature(order=2)

        for i, (xi, eta) in enumerate(gauss_points):
            N, dN_dxi_eta = shape_function_Q4(xi, eta)
            J, detJ, J_inv = compute_J_matrix(dN_dxi_eta, xy_coord)
            dN_dx_y = J_inv.dot(dN_dxi_eta)  # Convert to (x,y) derivatives
            B = compute_B_matrix(dN_dx_y)

            ke += np.dot(np.dot(B.T, C), B) * detJ * t * weights[i]

        return ke

    def apply_boundary_conditions(self, dirichlet_conditions):
        """
        Applies Dirichlet boundary conditions (prescribed displacements),
        allowing both zero and nonzero values.

        :param dirichlet_conditions: Dictionary specifying fixed nodes and their displacement values.
            Format: {node_id: ("X" or "Y" or "ALL", value)}
        :type dirichlet_conditions: dict
        """
        for node_id, (direction, displacement) in dirichlet_conditions.items():
            node = self.mesh.nodes[node_id]
            dof_indices = node.get_dof_indices()

            # Fix X-displacement
            if direction in ["X", "ALL"]:
                idx = dof_indices["X"]
                self.K_global[idx, :] = 0  # Zero out row
                self.K_global[:, idx] = 0  # Zero out column
                self.K_global[idx, idx] = 1  # Identity to prevent singularity
                self.F_global[idx] = displacement  # Set prescribed displacement

            # Fix Y-displacement
            if direction in ["Y", "ALL"]:
                idx = dof_indices["Y"]
                self.K_global[idx, :] = 0
                self.K_global[:, idx] = 0
                self.K_global[idx, idx] = 1
                self.F_global[idx] = displacement

    def apply_force(self, node_id, direction, force_value):
        """
        Applies an external force to a node.

        :param node_id: Node ID.
        :type node_id: int
        :param direction: Direction of force application ("X" or "Y").
        :type direction: str
        :param force_value: Magnitude of the applied force.
        :type force_value: float
        """
        dof_indices = self.mesh.nodes[node_id].get_dof_indices()
        self.F_global[dof_indices[direction]] += force_value

    def solve(self) -> np.ndarray:
        """
        Solves for nodal displacements.

        :return: The displacement vector containing nodal displacements.
        :rtype: np.ndarray
        """
        self.U = np.linalg.solve(self.K_global, self.F_global)
        return self.U

    def compute_stresses(self):
        """
        Computes element stresses based on solved displacements.

        :return: A dictionary mapping element IDs to their stress values.
        :rtype: dict
        """
        stress_results = {}
        for element in self.mesh.elements:
            dof_indices = element.get_dof_indices()
            element_displacements = self.U[dof_indices]

            xy_coord = np.array(element.get_node_coordinates())
            xi, eta = 0, 0  # Evaluating at the element center
            _, dN_dxi_eta = shape_function_Q4(xi, eta)
            J, detJ, J_inv = compute_J_matrix(dN_dxi_eta, xy_coord)
            dN_dx_y = J_inv.dot(dN_dxi_eta)
            B = compute_B_matrix(dN_dx_y)

            sigma = self.material.C.dot(B.dot(element_displacements))
            stress_results[element.element_id] = sigma

        return stress_results

    def plot_deformation(self, scale=10.0):
        """
        Plots the deformed shape of the structure, showing both original and displaced nodal coordinates.

        :param scale: Scaling factor for displacement visualization (default is 10.0).
        :type scale: float
        """
        if self.U is None or len(self.U) == 0:
            raise ValueError("No displacement data available. Solve the system first.")

        # Extract displacement components
        U_x = self.U[0::2]  # X-displacements
        U_y = self.U[1::2]  # Y-displacements

        # Get original nodal coordinates
        original_nodes = np.array([[node.coordinates['X'], node.coordinates['Y']] for node in self.mesh.nodes])

        # Compute displaced coordinates (scaled for visualization)
        displaced_nodes = original_nodes + scale * np.column_stack((U_x, U_y))

        # Initialize figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot original structure (light gray dashed line)
        for element in self.mesh.elements:
            x_coords = [node.coordinates['X'] for node in element.nodes] + [element.nodes[0].coordinates['X']]
            y_coords = [node.coordinates['Y'] for node in element.nodes] + [element.nodes[0].coordinates['Y']]
            ax.plot(x_coords, y_coords, color='gray', linestyle='--', linewidth=1.0)

        # Plot deformed structure (solid red line)
        for element in self.mesh.elements:
            deformed_x = [displaced_nodes[node.node_id, 0] for node in element.nodes] + [
                displaced_nodes[element.nodes[0].node_id, 0]]
            deformed_y = [displaced_nodes[node.node_id, 1] for node in element.nodes] + [
                displaced_nodes[element.nodes[0].node_id, 1]]
            ax.plot(deformed_x, deformed_y, color='red', linestyle='-', linewidth=1.5)

        # Plot nodes
        for node in self.mesh.nodes:
            ax.scatter(node.coordinates['X'], node.coordinates['Y'], color='blue', edgecolors='black', s=30, alpha=0.8)
            ax.scatter(displaced_nodes[node.node_id, 0], displaced_nodes[node.node_id, 1], color='red',
                       edgecolors='black', s=30, alpha=0.8)

        # Add grid and labels
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel("X [m]", fontsize=12, color='darkblue')
        ax.set_ylabel("Y [m]", fontsize=12, color='darkblue')
        ax.set_title(f"Deformed vs. Original Structure (Scale: {scale:.2f})", fontsize=14, fontweight='bold',
                     color='darkred')

        # Maintain aspect ratio and show the plot
        ax.set_aspect('equal')
        plt.show()

    def plot_stress(self, stress_type="sx"):
        """
        Plots the stress distribution as a filled contour plot, using nodal interpolation.

        The stress can be plotted based on different stress components:
            - "sx"  (Normal stress in X, σx)
            - "sy"  (Normal stress in Y, σy)
            - "txy" (Shear stress, τxy)
            - "svm" (Von Mises stress, σvm)

        :param stress_type: Type of stress to plot. Options: "sx", "sy", "txy", "svm". Default is "sx".
        :type stress_type: str
        """
        stress_index = {"sx": 0, "sy": 1, "txy": 2}

        if stress_type not in ["sx", "sy", "txy", "svm"]:
            raise ValueError("Invalid stress type. Choose from 'sx', 'sy', 'txy', or 'svm'.")

        if not self.U.any():
            raise ValueError("No displacement data available. Solve the system first.")

        stress_data = self.compute_stresses()  # Compute element stresses

        # Initialize stress values at nodes
        nodal_stresses = {node.node_id: [] for node in self.mesh.nodes}

        # Assign stress values from elements to nodes (averaging for shared nodes)
        for element in self.mesh.elements:
            element_stress = stress_data[element.element_id]

            # Compute Von Mises stress for each element
            sx, sy, txy = element_stress  # Extract stress components
            svm = np.sqrt(sx ** 2 + sy ** 2 - sx * sy + 3 * txy ** 2)  # Von Mises stress formula

            # Choose which stress to store
            if stress_type == "svm":
                stress_value = svm
            else:
                stress_value = element_stress[stress_index[stress_type]]

            for node in element.nodes:
                nodal_stresses[node.node_id].append(stress_value)

        # Compute averaged stress values at each node
        node_ids = np.array(list(nodal_stresses.keys()))
        node_coordinates = np.array([[node.coordinates['X'], node.coordinates['Y']] for node in self.mesh.nodes])
        node_stress_values = np.array([np.mean(nodal_stresses[nid]) for nid in node_ids])  # Averaging

        # Triangulation for smooth contour plot
        triangulation = tri.Triangulation(node_coordinates[:, 0], node_coordinates[:, 1])

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.tricontourf(triangulation, node_stress_values, cmap="coolwarm", levels=20)

        # Add colorbar
        cbar = fig.colorbar(contour)
        cbar.set_label(f"{stress_type.upper()} Stress [Pa]", fontsize=12)

        # Plot element edges for reference
        for element in self.mesh.elements:
            x_coords = [node.coordinates['X'] for node in element.nodes] + [element.nodes[0].coordinates['X']]
            y_coords = [node.coordinates['Y'] for node in element.nodes] + [element.nodes[0].coordinates['Y']]
            ax.plot(x_coords, y_coords, 'k-', linewidth=0.8, alpha=0.5)

        # Labels and formatting
        ax.set_xlabel("X [m]", fontsize=12, color='darkblue')
        ax.set_ylabel("Y [m]", fontsize=12, color='darkblue')
        ax.set_title(f"{stress_type.upper()} Stress Distribution", fontsize=14, fontweight='bold', color='darkred')

        ax.set_aspect('equal')
        plt.show()


    def __repr__(self):
        """
        Returns a string representation of the Solver object.

        :return: A formatted string showing solver properties.
        :rtype: str
        """
        return f"Solver(FEM model with {len(self.mesh.nodes)} nodes and {len(self.mesh.elements)} elements)"
