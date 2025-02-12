import numpy as np

class Material:
    """
    Represents the material properties for FEM analysis,
    supporting both Plane Stress and Plane Strain conditions.

    :param name: Name of the material.
    :type name: str
    :param E: Young's Modulus (Pa).
    :type E: float
    :param poisson_ratio: Poisson's ratio (-).
    :type poisson_ratio: float
    :param stress_state: Specifies the analysis type. Options are "Plane Stress" or "Plane Strain".
                         Defaults to "Plane Stress".
    :type stress_state: str
    """

    def __init__(self, name: str, E: float, poisson_ratio: float, stress_state: str = "Plane Stress"):

        self.name = name
        self.E = E
        self.poisson_ratio = poisson_ratio
        self.stress_state = stress_state  # "Plane Stress" or "Plane Strain"
        self.C = self._compute_C_matrix()  # Compute the material stiffness matrix

    def _compute_C_matrix(self) -> np.ndarray:
        """
        Computes the constitutive (stiffness) matrix based on the selected stress state.

        :return: The 3x3 stiffness matrix (`C matrix`) used in FEM calculations.
        :rtype: np.ndarray
        """
        nu = self.poisson_ratio
        E = self.E

        if self.stress_state.lower() == "plane stress":
            # Plane Stress elasticity matrix (3x3)
            factor = E / (1 - nu ** 2)
            C = factor * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2]
            ])
        elif self.stress_state.lower() == "plane strain":
            # Plane Strain elasticity matrix (3x3)
            factor = E / ((1 + nu) * (1 - 2 * nu))
            C = factor * np.array([
                [1 - nu, nu, 0],
                [nu, 1 - nu, 0],
                [0, 0, (1 - 2 * nu) / 2]
            ])
        else:
            raise ValueError("Invalid stress state. Choose 'Plane Stress' or 'Plane Strain'.")

        return C

    def __repr__(self):
        """
        Returns a string representation of the Material object.

        :return: A formatted string showing material properties.
        :rtype: str
        """
        return (f"Material(Name={self.name}, E={self.E:.2e} Pa, Poisson={self.poisson_ratio}, "
                f"Stress State={self.stress_state}")
