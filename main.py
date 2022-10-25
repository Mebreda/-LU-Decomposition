import sys
from sympy.solvers import solve
from sympy import Symbol
import numpy as np


def input_values():
    degree = int(input("Enter the degree of your polynomial: "))
    variables = ['x', 'y', 'z', 'w', 't', 'u', 'v']

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
    # print("upper ", U)
    # print("lower ", L)
    return U, L


def forward_substitution(L, B):
    D = {"d1": 0, "d2": 0, "d3": 0, "d4": 0, "d5": 0, "d6": 0, "d7": 0}
    n = len(L)
    i = 1
    j = 0
    equation = ""
    D["d1"] = B[0][0]
    while i < n:
        while L[i][j] != 0:
            equation = equation + str(L[i][j]) + " * " + str(D["d" + str(j + 1)])
            if L[i][j] > 0 and L[i][j + 1] != 0:
                equation = equation + " + "

            j += 1
        value = 0
        j = 0
        i += 1


if __name__ == "__main__":
    x = 2
    y = -1
    s = solve(x+y)
    print(s[0])
    # A, B, X = input_values()
    A = np.array([[2.0, -2.0, 4.0, 6.0], [2.0, 3.0, -4.0, -1.0], [-1.0, 2.0, -5.0, -4.0], [3.0, 2.0, 3.0, 7.0]])
    # A = np.array([[1, -1, 2,], [2, 3, -4, -1], [-1, 2, -5, -4], [3, 2, 3, 7]])
    B = np.array([[6, -2, 7, 4]])
    U, L = decomposition(A)
    forward_substitution(L, B)
