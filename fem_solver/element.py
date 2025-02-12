class Element:
    """Represents an element in an FEM mesh, defined by its nodes."""

    def __init__(self, element_id: int, nodes, element_type="Quadrilateral 4-node"):
        """
        Initializes an Element object.

        Parameters:
        - element_id (int): Unique identifier for the element.
        - nodes (list[Node]): List of Node objects that define the element.
        - element_type (str): Type of element (default is "Quadrilateral 4-node").
        """
        if len(nodes) < 3:
            raise ValueError("An element must have at least 3 nodes.")

        self.element_id = element_id
        self.nodes = nodes  # List of Node objects
        self.element_type = element_type

    def get_dof_indices(self):
        """Returns the global DOF indices for all nodes in the element."""
        return [dof for node in self.nodes for dof in node.get_dof_indices().values()]

    def get_node_coordinates(self):
        """Returns the coordinates of the element's nodes as a list of (x, y) tuples."""
        return [(node.coordinates['X'], node.coordinates['Y']) for node in self.nodes]

    def __repr__(self):
        node_ids = [node.node_id for node in self.nodes]
        return (f"Element(ID={self.element_id}, Type={self.element_type}, Nodes={node_ids}, "
                f"DOFs={self.get_dof_indices()}, Coordinates={self.get_node_coordinates()})")
