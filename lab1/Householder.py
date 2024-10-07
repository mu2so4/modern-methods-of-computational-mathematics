import numpy as np

def HouseholderQrDecomposition(matrix: np.matrix):
    rowCount, columnCount = matrix.shape
    R = matrix.copy()
    reflections = []
    
    for iteration in range(0, columnCount):
        subsize = rowCount - iteration

        e = np.zeros(subsize)
        e[0] = 1

        x = R[iteration:, iteration]

        normX = np.linalg.norm(x)
        if normX == 0:
            raise Exception("Norm cannot be equal to 0")
        
        u0 = x + np.sign(x[0]) * normX * e

        u = u0 / np.linalg.norm(u0)
        u = u[np.newaxis]

        UpdateR(R, u, iteration)
        reflections.append(u)

    R = np.triu(R)
    Q = np.eye(rowCount)
    for i, u in reversed(list(enumerate(reflections))):
        UpdateR(Q, u, i)

    return Q, R

def UpdateR(M, u, iteration):
    Mt = M[iteration:, iteration:]
    M[iteration:, iteration:] = Mt - (2 * u.T) @ (u @ Mt)
    pass