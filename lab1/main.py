import numpy as np
import time
import LeastSquares
import Householder

def ReadInput(filename) -> np.ndarray | np.ndarray:
    epsilonList = []
    sigmaList = []
    with open(filename) as file:
        for line in file:
            epsilon, sigma = map(float, line.split())
            epsilonList.append(epsilon)
            sigmaList.append(sigma)
    return np.array(epsilonList), np.array(sigmaList)

def ChebyshevPolynomial(value, count) -> np.array:
    arr = np.zeros(count)
    arr[0] = 1
    arr[1] = value

    for index in range(2, count):
        arr[index] = 2 * value * arr[index - 1] - arr[index - 2]

    return arr

def ChebyshevPolynomialMatrix(values, count) -> np.array:
    return np.array([ChebyshevPolynomial(v, count) for v in values])

epsilon, sigma = ReadInput("lab1/2.txt")


limit = 2000
epsilon = epsilon[:limit]
sigma = sigma[:limit]


polynomialCount = 6

chebyshevMatrix = ChebyshevPolynomialMatrix(epsilon, polynomialCount)

cond = np.linalg.cond(chebyshevMatrix)
print(f'Conditional number of A: {cond}')

startTime = time.process_time()
#coefficients = LeastSquares.NormalEquations(chebyshevMatrix, sigma)
coefficients = LeastSquares.QrDecomposition(chebyshevMatrix, sigma, Householder.HouseholderQrDecomposition)
endTime = time.process_time()

print(f'elapsed time {endTime - startTime} s')
print(coefficients)

'''
print('Epsilon\tSigma_e\tSigma_a\tDiff')
for index in range(0, 10):
    currentEpsilon = epsilon[index]
    currentSigmaExpected = sigma[index]
    currentSigmaActual = np.polynomial.chebyshev.chebval(currentEpsilon, coefficients)
    diff = currentSigmaActual - currentSigmaExpected
    print(f'{currentEpsilon}\t{currentSigmaExpected}\t{currentSigmaActual}\t{diff}')
'''
