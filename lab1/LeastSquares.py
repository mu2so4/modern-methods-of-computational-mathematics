import numpy as np
from scipy.linalg import cholesky, solve_triangular


def NormalEquations(matrix, b) -> np.array:
    transposed = matrix.T
    # the solution is x = (A^T A)^-1 A^T b
    matProd = transposed @ matrix
    choleskyLower = cholesky(matProd, lower=True)

    # solve the system Cy=b', C^T x=y, where C is a lower triangular Cholesky matrix, b'=C^T b
    bConverted = transposed @ b
    y = solve_triangular(choleskyLower, bConverted, lower=True)
    x = solve_triangular(choleskyLower.T, y, lower=False)
    return x


def QrDecomposition(matrix, b, DecomposeQR) -> np.array:
    #to QRx=b
    Q, R = DecomposeQR(matrix)
    # using the statement that Q^-1=Q^T we convert the expression to Rx=Q^T b
    bConverted = np.transpose(Q) @ b
    print(Q.shape)
    print(R.shape)
    print(bConverted.shape)
    x = sc.linalg.solve_triangular(R, bConverted)
    return x
