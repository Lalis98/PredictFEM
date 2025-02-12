import numpy as np

def shape_function_Q4(xi_, eta_):

    N1 = 1 / 4 * (1 - xi_) * (1 - eta_)
    N2 = 1 / 4 * (1 + xi_) * (1 - eta_)
    N3 = 1 / 4 * (1 + xi_) * (1 + eta_)
    N4 = 1 / 4 * (1 - xi_) * (1 + eta_)

    dN1 = 1 / 4 * np.array([-(1 - eta_), -(1 - xi_)])
    dN2 = 1 / 4 * np.array([1 - eta_, -(1 + xi_)])
    dN3 = 1 / 4 * np.array([1 + eta_, 1 + xi_])
    dN4 = 1 / 4 * np.array([-(1 + eta_), (1 - xi_)])

    N_ = np.array([
        [N1],
        [N2],
        [N3],
        [N4]
    ])

    dN_ = np.array([
        dN1,
        dN2,
        dN3,
        dN4
    ])

    return N_, dN_.T


def gauss_quadrature(order: int = 2):

    if order == 2:

        # Gauss-Points Coordinates of (ξ,η)
        xi_eta = np.array([
            [-np.sqrt(1 / 3), -np.sqrt(1 / 3)],
            [np.sqrt(1 / 3), -np.sqrt(1 / 3)],
            [-np.sqrt(1 / 3), np.sqrt(1 / 3)],
            [np.sqrt(1 / 3), np.sqrt(1 / 3)]])

        w = np.array([1, 1, 1, 1])  # Weight of Gauss-Points

    else:
        xi_eta = np.array([[0, 0]])
        w = np.array([4])

    return xi_eta, w


def compute_J_matrix(natural_derivative, xy_coordinates):
    J = natural_derivative @ xy_coordinates  # Matrix multiplication
    detJ = np.linalg.det(J)  # Determinant of Jacobian
    invJ = np.linalg.inv(J)  # Inverse of Jacobian
    return J, detJ, invJ


def compute_B_matrix(dN):
    """Compute strain-displacement (B) matrix at Gauss points."""

    B_matrix = np.zeros((3, 8))  # Creating Matrix B 3x8

    B_matrix[0, 0::2] = dN[0, :]  # row 1 of Matrix B
    B_matrix[1, 1::2] = dN[1, :]  # row 2 of Matrix B
    B_matrix[2, 0::2] = dN[1, :]  # row 3, odd columns of Matrix B
    B_matrix[2, 1::2] = dN[0, :]  # row 3, even columns of Matrix B

    return B_matrix
