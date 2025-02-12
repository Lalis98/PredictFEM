class DOF:
    """
    Represents the degrees of freedom (DOFs) for a node in an FEM analysis.

    :param node_id: Node ID number.
    :type node_id: int

    """

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.dof_x = 2 * node_id      # Global DOF index for x-displacement
        self.dof_y = 2 * node_id + 1  # Global DOF index for y-displacement

        self.displacement = {"X": None, "Y": None} # Dictionary to store prescribed displacements (None means it is free)
        self.force = {"X": 0.0, "Y": 0.0} # Forces applied at this node

    def fix(self, direction: str, value: float = 0.0):
        """
        Fixes the DOF in a given direction ("X", "Y", or "ALL") at a prescribed displacement value.

        :param direction: Fix the "X", "Y" or "ALL" DOFs.
        :type direction: str
        :param value: The prescribed displacement value. Defaults to 0.0.
        :type value: float
        """

        direction = direction.upper()  # Make all characters upper-case
        if direction in self.displacement:
            self.displacement[direction] = value  # If direction == "X" or "Y"
        elif direction == "ALL":  # If direction == "ALL"
            self.displacement["X"] = value
            self.displacement["Y"] = value


    def apply_force(self, direction: str, value: float):
        """
        Applies force in the given direction ("X" or "Y").

        :param direction: Apply Force to "X" or "Y".
        :type direction: str
        :param value: The applied force at the specific direction. Defaults to 0.0.
        :type value: float
        """

        direction = direction.upper()
        if direction in self.force:
            self.force[direction] += value
        else:
            raise ValueError("Invalid force direction. Use 'X' or 'Y'.")

    def is_fixed(self, direction: str) -> bool:
        """
        Checks if a DOF is fixed.

        :param direction: The direction to check ("X" or "Y").
        :type direction: str

        :return type: bool
        """
        return self.displacement[direction] is not None

    def get_dof_indices(self):
        """
        Returns the global DOF indices for the node.

        :return: A dictionary containing the DOF indices.
        :rtype: dict
        """
        return {"X": self.dof_x, "Y": self.dof_y}

    def __repr__(self):
        """
        Returns a string representation of the DOF object.

        :return: Returns a string representation of the DOF object.
        :rtype: str
        """
        return (f"DOF(Node={self.node_id}, DOFs={{'X': {self.dof_x}, 'Y': {self.dof_y}}}, "
                f"Fixed={self.displacement}, Forces={self.force})")
