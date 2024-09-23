import numpy as np
import LeastSquares
import Householder


'''
Pseudocode

1. Read data in format (epsilon_i, sigma_i), where i in range(0, M)

I have Chebyshev polynomials. Count of polynomials N < M - 1.

T_0(epsilon) = 1, T_1(epsilon) = epsilon,
T_{n+1}(epsilon) = 2 * epsilon * T_n(epsilon) - T_{n-1}(epsilon)

T_i(epsilon_j) are CONSTANTS. I need a function to evaluate them.

'''

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

def EvaluateChebyshevPolynomial(value: float, coefs) -> float:
    if len(coefs) == 0:
        return 0.0
    elif len(coefs) == 1:
        return coefs[0]

    prevPrev = 1
    prev = value
    result = prevPrev * coefs[0]

    if len(coefs) > 1:
        result += coefs[1] * prev

    for index in range(2, len(coefs)):
        current = 2 * value * prev - prevPrev
        result += coefs[index] * current
        prevPrev = prev
        prev = current
    return result

def ChebyshevPolynomialMatrix(values, count) -> np.array:
    return np.array([ChebyshevPolynomial(v, count) for v in values])

epsilon, sigma = ReadInput("2.txt")

'''
limit = 1500
epsilon = epsilon[:limit]
sigma = sigma[:limit]
'''

polynomialCount = 7

chebyshevMatrix = ChebyshevPolynomialMatrix(epsilon, polynomialCount)

cond = np.linalg.cond(chebyshevMatrix)
print(f'Conditional number of A: {cond}')

coefficientsNormal = LeastSquares.NormalEquations(chebyshevMatrix, sigma)
#coefficientsNormal = LeastSquares.QrDecomposition(chebyshevMatrix, sigma, Householder.HouseholderQrDecomposition)

print(coefficientsNormal)

print('Epsilon\tSigma_e\tSigma_a\tDiff')
for index in range(0, 200):
    currentEpsilon = epsilon[index]
    currentSigmaExpected = sigma[index]
    #currentSigmaActual = np.polynomial.chebyshev.chebval(currentEpsilon, coefficientsNormal)
    currentSigmaActual = EvaluateChebyshevPolynomial(currentEpsilon, coefficientsNormal)
    diff = currentSigmaActual - currentSigmaExpected
    print(f'{currentEpsilon}\t{currentSigmaExpected}\t{currentSigmaActual}\t{diff}')

