from .dof import DOF

class Node:
    """
    Represents a node in an FEM mesh, including its coordinates and DOF.

    :param node_id: Unique identifier for the node.
    :type node_id: int
    :param x: X-coordinate of the node.
    :type x: float
    :param y: Y-coordinate of the node.
    :type y: float
    """

    def __init__(self, node_id: int, x: float, y: float):

        self.node_id = node_id
        self.coordinates = {"X": x, "Y": y}  # Store coordinates in a dictionary for clarity
        self.dof = DOF(node_id)  # Each node has an associated DOF object

    def fix(self, direction: str, value: float = 0.0):
        """
        Fixes the node's displacement in the given direction.

        :param direction: Direction to fix ("X", "Y", or "ALL").
        :type direction: str
        :param value: Prescribed displacement value. Defaults to 0.0.
        :type value: float
        """
        self.dof.fix(direction, value)

    def apply_force(self, direction: str, value: float):
        """
        Applies force to the node in the specified direction.

        :param direction: Direction to apply force ("X" or "Y").
        :type direction: str
        :param value: Force magnitude.
        :type value: float
        """
        self.dof.apply_force(direction, value)

    def get_dof_indices(self):
        """
        Returns the global DOF indices associated with the node.

        :return: A dictionary containing the DOF indices.
        :rtype: dict
        """
        return self.dof.get_dof_indices()

    def is_fixed(self, direction: str) -> bool:
        """
        Checks if the node is fixed in a given direction.

        :param direction: The direction to check ("X" or "Y").
        :type direction: str

        :return: True if the DOF is fixed, otherwise `False`.
        :rtype: bool
        """
        return self.dof.is_fixed(direction)


    def __repr__(self):
        """
        Returns a string representation of the Node object.

        :return: Returns a string representation of the node object.
        :rtype: str
        """
        return (f"Node(ID={self.node_id}, Coordinates={self.coordinates}, "
                f"DOFs={self.get_dof_indices()}, Fixed={self.dof.displacement}, "
                f"Forces={self.dof.force})")
