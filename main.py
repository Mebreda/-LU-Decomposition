from sympy.solvers import solve
from sympy import Symbol
import numpy as np


def input_values():
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    degree = int(input("Enter the degree of your polynomial: "))
    variables = []
    for i in range(degree):
        variables.append(('x' + str(i + 1)).translate(sub))

    print(variables)
    c = 0
    c1 = 0
    coefficient_array = []
    B = None
    X = None
    A = None
    print("Enter the coefficient of the following variables")
    while degree > c:
        print("Equation ", c + 1)
        while degree > c1:
            coefficient = float(input(variables[c1] + "= "))
            coefficient_array.append(coefficient)
            if X is None:
                X = np.array([variables[c1]])
            else:
                if variables[c1] not in X:
                    X = np.append(X, [variables[c1]], axis=0)

            c1 += 1
        # constant = float(input("c = "))
        # coefficient_array.append(constant)
        print("Enter the value of the sum")
        s = float(input("sum = "))
        if B is None:
            B = np.array([s])
        else:
            B = np.append(B, [s], axis=0)

        if A is None:
            A = np.array([coefficient_array])
        else:
            A = np.append(A, [coefficient_array], axis=0)

        coefficient_array = []
        c1 = 0
        c += 1
    return A, B, X


def decomposition(A):
    n = len(A)
    print(n)
    U = np.copy(A)
    L = np.copy(A)
    i = 0
    j = 0
    while i < n:
        while j < n - 1:
            j += 1
            L[i][j] = 0

        L[i][i] = 1
        i += 1
        j = i

    i = 0
    k = 0
    while i < n:

        while k < n - 1:

            divisor = U[i][i]
            k += 1
            j = 0
            if U[k][i] > 0:
                multiplier = -abs(U[k][i])
            else:
                multiplier = abs(U[k][i])

            L[k][i] = U[k][i] / divisor
            while j < n:
                U[k][j] = (U[i][j] / divisor) * multiplier + U[k][j]
                j += 1
        i += 1
        k = i
    print("upper ", U)
    print("lower ", L)
    return U, L


def forward_substitution(L, B):
    D = {}
    n = len(L)
    i = 0
    while i < n:
        D.update({"d" + str(i + 1): 0})
        i += 1

    i = 1
    j = 0
    equation = 0
    D["d1"] = B[0]

    while i < n:
        while j < n:
            if D["d" + str(j + 1)] == 0:
                break
            equation = equation + (L[i][j] * D["d" + str(j + 1)])
            j += 1

        x = Symbol('x')
        value = solve(equation + x - B[j])

        D["d" + str(i + 1)] = value[0]
        j = 0
        i += 1
        equation = 0

    D_array = np.array([D["d1"]])
    i = 1
    while i < len(D):
        D_array = np.append(D_array, [D["d" + str(i + 1)]], axis=0)
        i += 1
    return D_array

# def backward_substitution(U, D, X):
#     n = len(U)
#
#     x = np.zeros_like(D)
#
#     x[-1] = D[-1] / U[-1, -1]
#
#     for i in range(n - 2, -1, -1):
#         x[i] = (D[i] - np.dot(U[i, i:], x[i:])) / U[i, i]
#
#     return x

def backward_substitution(U, D):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        tmp = D[i]
        for j in range(i + 1, n):
            tmp -= U[i, j] * x[j]
        x[i] = tmp / U[i, i]
    return x



if __name__ == "__main__":
    # A, B = input_values()

    # for testing purposes
    A = np.array([[1.0, -1.0, 2.0, 3.0], [2.0, 3.0, -4.0, -1.0], [-1.0, 2.0, -5.0, -4.0], [3.0, 2.0, 3.0, 7.0]])
    B = np.array([6, -2, -7, 4])

    U, L = decomposition(A)
    D = forward_substitution(L, B)
    M = backward_substitution(U, D)
    print(" ")
    
    # for testing purposes
    print("D = ", D)
    print(" ")
    counter = 0
    x_balyu = len(M)
    while counter < x_balyu:
        print("X", counter + 1, " = ", round(M[counter]))
        counter += 1

    # print("M = ", round(M[0]), round(M[1]), round(M[2]), round(M[3]))

