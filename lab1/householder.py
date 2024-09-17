import numpy as np

def HouseholderDecomposition(matrix: np.matrix) -> np.matrix | np.matrix:
    rowCount =
    columnCount = 
    # columnCount < rowCount
    rMatrix = matrix
    for iteration in range(0, columnCount):
        '''
        rows and columns of the matrix are numbered from 0.
        Steps for i iteration:

        e = (1 0 ... 0)^T, dim(e) = rowCount - i

        x = (a_{i, i} ... a_{i, rowColumn - 1}), dim(x) = rowColumn - i
        in other words, x is a vector created by elements on i-th column
        which are on and under diagonal.
        
        u_0 = x + sgn(x_0) * ||x||_2 * e, where sgn(x_0) is the sign
        function, ||x||_2 is the euclidian norm.

        u = u_0 / ||u_0|| -- it's normalizing

        H_i' = E - 2 * u * u^T, where E is the identity matrix with size
        (rowCount - i).

        H_i =
        |E   0  |
        |0   H_i|

        R_i = (i == 0) ? H_0 * A : H_i * R_{i-1}
        Q_i = (i == 0) ? H_0 : H_i * Q_{i-1}
        
        OUTPUT:
        Q = Q_{rowCount-1}
        R = R_{rowCount-1}
        '''
    return None, rMatrix
