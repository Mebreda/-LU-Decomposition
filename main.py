import sys

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
        constant = float(input("c = "))
        coefficient_array.append(constant)
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
    U = A
    i = 0
    j = 0
    k = 0
    while i < n:

        while k < n-1:

            divisor = U[i][i]
            #while j < n:
            #    L[i][j] = L[i][j] / divisor
            #    j += 1

            k += 1
            j = 0
            if U[k][i] > 0:
                multiplier = -abs(U[k][i])
            else:
                multiplier = abs(U[k][i])

            while j < n:
                U[k][j] = (U[i][j]/divisor) * multiplier + U[k][j]
                j += 1
        j = 0


        i += 1
        k = i
    print(U)



if __name__ == "__main__":
    # A, B, X = input_values()
    A = np.array([[2.0, -2.0, 4.0, 6.0], [2.0, 3.0, -4.0, -1.0], [-1.0, 2.0, -5.0, -4.0], [3.0, 2.0, 3.0, 7.0]])
    # A = np.array([[1, -1, 2,], [2, 3, -4, -1], [-1, 2, -5, -4], [3, 2, 3, 7]])
    decomposition(A)

    # print(len(A))
