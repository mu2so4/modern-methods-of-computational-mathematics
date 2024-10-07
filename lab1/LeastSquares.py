import numpy as np
from scipy.linalg import solve_triangular


def NormalEquations(matrix, b) -> np.array:
    transposed = matrix.T
    # the solution is x = (A^T A)^-1 A^T b
    matProd = transposed @ matrix

    bConverted = transposed @ b
    return np.linalg.solve(matProd, bConverted)


def QrDecomposition(matrix, b, DecomposeQR) -> np.array:
    #to QRx=b
    Q, R = DecomposeQR(matrix)
    # using the statement that Q^-1=Q^T we convert the expression to Rx=Q^T b
    Qtb = Q.T @ b

    _, n = R.shape
    
    # Truncate R to square form if the system is overdetermined (m > n)
    R_square = R[:n, :n]  # Select the upper square part of R
    b_truncated = Qtb[:n]  # Truncate b accordingly

    x = solve_triangular(R_square, b_truncated, lower=False)
    return x
