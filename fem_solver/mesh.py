import numpy as np
import matplotlib.pyplot as plt

from .node import Node
from .element import Element

class Mesh:
    """Represents a mesh object for FEM analysis."""

    def __init__(self):
        """Initializes an empty mesh structure."""
        self.nodes = []      # List of all nodes
        self.elements = []   # List of all elements
        self.node_map = {}   # Dictionary to store (i, j) → node reference

        self.num_x = None
        self.num_y = None
        self.length = None
        self.height = None
        self.thickness = None  # Added thickness attribute

    def create_rectangular_mesh(self, num_x: int, num_y: int, length: float, height: float, thickness: float):
        """
        Generates a structured rectangular mesh.

        :param num_x: Number of nodes in the X direction.
        :type num_x: int
        :param num_y: Number of nodes in the Y direction.
        :type num_y: int
        :param length: Total length of the rectangular domain (m).
        :type length: float
        :param height: Total height of the rectangular domain (m).
        :type height: float
        :param thickness: Thickness (m) of the mesh (for later use in FEM solver).
        :type thickness: float
        """
        self.num_x = num_x
        self.num_y = num_y
        self.length = length
        self.height = height
        self.thickness = thickness  # Store thickness

        self.nodes = []  # Reset lists in case function is called multiple times
        self.elements = []
        self.node_map = {}

        self._generate_nodes()
        self._generate_elements()

    def _generate_nodes(self):
        """
        Creates the nodes of the mesh in a structured grid.
        """
        x_coords = np.linspace(0.0, self.length, self.num_x)
        y_coords = np.linspace(0.0, self.height, self.num_y)

        node_id = 0
        for j, y in enumerate(y_coords):
            for i, x in enumerate(x_coords):
                node = Node(node_id, x, y)
                self.nodes.append(node)
                self.node_map[(i, j)] = node  # Store node reference
                node_id += 1

    def _generate_elements(self):
        """
        Creates quadrilateral elements using the generated nodes.
        """
        element_id = 0
        for j in range(self.num_y - 1):
            for i in range(self.num_x - 1):
                n1 = self.node_map[(i, j)]
                n2 = self.node_map[(i + 1, j)]
                n3 = self.node_map[(i + 1, j + 1)]
                n4 = self.node_map[(i, j + 1)]

                element = Element(element_id, [n1, n2, n3, n4], "Quadrilateral 4-node")
                self.elements.append(element)
                element_id += 1


    def plot_mesh(self):
        """
        Visualizes the generated mesh with nodes and elements.
        """
        if not self.nodes or not self.elements:
            raise ValueError("No mesh has been created. Call `create_rectangular_mesh()` first.")

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot elements with a soft gray dashed line
        for element in self.elements:
            x_coords = [node.coordinates['X'] for node in element.nodes] + [element.nodes[0].coordinates['X']]
            y_coords = [node.coordinates['Y'] for node in element.nodes] + [element.nodes[0].coordinates['Y']]
            ax.plot(x_coords, y_coords, color='gray', linestyle='--', linewidth=1.0)

        # Plot nodes with red markers
        for node in self.nodes:
            ax.scatter(node.coordinates['X'], node.coordinates['Y'], color='red', edgecolors='black', s=30, alpha=0.8)
            ax.text(node.coordinates['X'], node.coordinates['Y'], f'{node.node_id}', fontsize=9, color='black',
                    verticalalignment='bottom', horizontalalignment='right')

        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5) # Add grid for better visibility

        # Adjust axes limits with a small margin
        ax.set_xlim(-0.1, self.length + 0.1)
        ax.set_ylim(-0.1, self.height + 0.1)

        # Titles and labels
        ax.set_xlabel("X [m]", fontsize=12, color='darkblue')
        ax.set_ylabel("Y [m]", fontsize=12, color='darkblue')
        ax.set_title(f"Finite Element Mesh (Thickness = {self.thickness} m)", fontsize=14, fontweight='bold',
                     color='darkred')

        ax.set_aspect('equal') # Maintain equal aspect ratio

        plt.show()

    def __repr__(self) -> str:
        """
        Returns a string representation of the Mesh object.

        :return: A formatted string showing mesh properties.
        :rtype: str
        """

        return (f"Mesh(Num Nodes={len(self.nodes)}, Num Elements={len(self.elements)}, "
                f"Size=({self.length} x {self.height}), Grid=({self.num_x} x {self.num_y}), "
                f"Thickness={self.thickness})")
