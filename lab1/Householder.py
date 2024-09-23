import numpy as np

def HouseholderQrDecomposition(matrix):
    rowCount, columnCount = matrix.shape
    rMatrix = matrix
    print(columnCount)
    for iteration in range(0, columnCount):
        subsize = rowCount - iteration

        e = np.zeros(subsize)
        e[0] = 1

        x = np.array([rMatrix[row, iteration] for row in range(iteration, rowCount)])
        u0 = x + (1 if x[0] > 0 else -1) * np.linalg.norm(x) * e

        u = u0 / np.linalg.norm(u0)
        # prepare for transposing
        u = u[np.newaxis]

        HiPart = np.eye(subsize) - 2 * u * u.T
	
        Hi = MergeMatrices(np.eye(iteration), HiPart) if iteration > 0 else HiPart

        rMatrix = np.matmul(Hi, rMatrix)
        
        if iteration == 0:
            qMatrix = Hi
        else:
            qMatrix = np.matmul(Hi, qMatrix)

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
