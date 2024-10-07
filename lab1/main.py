import numpy as np
import time
from matplotlib import pyplot as plt
import LeastSquares
import Householder
import sys

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
    if count == 0:
        raise Exception("count must be greater than 0")
    if count == 1:
        return np.array([1])
    
    arr = np.zeros(count)
    arr[0] = 1
    arr[1] = value

    for index in range(2, count):
        arr[index] = 2 * value * arr[index - 1] - arr[index - 2]

    return arr

def ChebyshevPolynomialMatrix(values, count) -> np.array:
    return np.array([ChebyshevPolynomial(v, count) for v in values])


def plot_graphics(epsilon, sigma, approximation_data, n):
    fig = plt.figure(figsize=(12,8), frameon=True)
    plt.style.use('ggplot')
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['font.size'] = 20
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    ax = fig.add_subplot(111)    

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')

    ax.set(facecolor='w')
    ax.grid('axis = "both"', color = 'gray')

    ax.set_xlabel('$x$', labelpad = -10)
    ax.set_ylabel('$y$', rotation = 0, labelpad = 20)

    ax.plot(epsilon, sigma, color = 'blue', linestyle = '-', linewidth = 3, label='Data')
    #ax.plot(w_obr, P_obr, color = 'red', linestyle = '-', label = 'Processed')
    ax.plot(epsilon, approximation_data, color = 'red', linestyle = '-', linewidth = 2, label = 'Approximation with N = ' + str(n))
    ax.legend(loc="upper left")

    plt.show()

def nrmse(expected, actual) -> float:
    if len(expected) != len(actual):
        raise Exception("must be the same size")
    count = len(expected)
    maxExp = max(expected)
    squares = sum([(actual[i] - expected[i]) ** 2 for i in range(0, count)])
    return np.sqrt(squares / count) / maxExp

if(len(sys.argv) != 3):
    print("Usage: {} filename N".format(sys.argv[0]))
    exit(1)

filename = sys.argv[1]
polynomialCount = int(sys.argv[2])

epsilon, sigma = ReadInput(filename)

chebyshevMatrix = ChebyshevPolynomialMatrix(epsilon, polynomialCount)

cond = np.linalg.cond(chebyshevMatrix)
#print(f'Conditional number of A: {cond}')


startTime = time.time()
coefficients = LeastSquares.NormalEquations(chebyshevMatrix, sigma)
#coefficients = LeastSquares.QrDecomposition(chebyshevMatrix, sigma, Householder.HouseholderQrDecomposition)
endTime = time.time()

print('elapsed time {:.2e} s'.format(endTime - startTime))
#print(coefficients)

apprSigma = [np.polynomial.chebyshev.chebval(eps, coefficients) for eps in epsilon]

mse = nrmse(sigma, apprSigma)

print("mu(AA^T)\tmu(A)\tMSE")
print('{:.2e}\t{:.2e}\t{:.2e}'.format(cond * cond, cond, mse))

#plot_graphics(epsilon, sigma, apprSigma, polynomialCount)

