import numpy as np

def HouseholderQrDecomposition(matrix):
    rowCount, columnCount = matrix.shape
    rMatrix = matrix.copy()
    qMatrix = np.eye(rowCount)
    
    for iteration in range(0, columnCount):
        subsize = rowCount - iteration

        e = np.zeros(subsize)
        e[0] = 1

        x = rMatrix[iteration:, iteration]

        normX = np.linalg.norm(x)
        if normX == 0:
            raise Exception("Norm cannot be equal to 0")
        
        u0 = x + np.sign(x[0]) * normX * e

        u = u0 / np.linalg.norm(u0)
        # prepare for transposing
        u = u[np.newaxis]

        HiPart = np.eye(subsize) - 2 * np.outer(u, u)
	
        Hi = MergeMatrices(np.eye(iteration), HiPart) if iteration > 0 else HiPart

        rMatrix = Hi @ rMatrix
        qMatrix = qMatrix @ Hi.T

    return qMatrix, rMatrix


def MergeMatrices(A, B):
    # Получаем размеры матриц A и B
    rowsA, columnsA = A.shape
    rowsB, columnsB = B.shape
    
    # Размер новой матрицы: строки — это сумма строк A и B, столбцы — сумма столбцов A и B
    mergedRows = rowsA + rowsB
    mergedColumns = columnsA + columnsB
    
    # Создаем нулевую матрицу необходимого размера
    mergedMatrix = np.zeros((mergedRows, mergedColumns), dtype=A.dtype)
    
    # Вставляем A в верхний левый угол
    mergedMatrix[:rowsA, :columnsA] = A
    
    # Вставляем B, начиная с позиции (rowsA, columnsA)
    mergedMatrix[rowsA:mergedRows, columnsA:mergedColumns] = B
    
    return mergedMatrix
