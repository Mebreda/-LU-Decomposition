from sympy.solvers import solve
from sympy import Symbol
import numpy as np

"""
S
Asks the user for inputs
    - number of variables
    - coefficients of each variable
    - value of each equation
Stores it to a NumPy matrix and returns the three matrix
    - A = coefficient matrix
    - B = matrix of numbers on the right-hand side of the equations
    - X = variable matrix
"""


def input_values():
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    degree = int(input("Enter the number of variables: "))
    variables = []
    # A loop that will save unique variables depending on the input
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
    # A loop that will ask for the values of the problem
    while degree > c:
        print("Equation ", c + 1)
        # Stores the coefficients, variables, and the value of the equation
        while degree > c1:
            coefficient = float(input(variables[c1] + "= "))
            coefficient_array.append(coefficient)
            if X is None:
                X = np.array([variables[c1]])
            else:
                if variables[c1] not in X:
                    X = np.append(X, [variables[c1]], axis=0)

            c1 += 1

        print("Enter the value of the equation ", c + 1)
        s = float(input("value = "))
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


"""
1st step(decomposition) of the LU Decomposition 
Applies the Gaussian Elimination to obtain the matrix U while simultaneously solving the values of matrix L
Returns matrix U and matrix L
"""


def decomposition(A):
    n = len(A)
    print(n)
    U = np.copy(A)
    L = np.copy(A)
    i = 0
    j = 0
    # Sets the needed 0s and 1s for the matrix L
    while i < n:
        while j < n - 1:
            j += 1
            L[i][j] = 0

        L[i][i] = 1
        i += 1
        j = i

    i = 0
    k = 0
    # Gaussian Elimination for matrix U
    # Solves the values per row
    while i < n:

        while k < n - 1:

            divisor = U[i][i]
            k += 1
            j = 0
            if U[k][i] > 0:
                multiplier = -abs(U[k][i])
            else:
                multiplier = abs(U[k][i])

            L[k][i] = U[k][i] / divisor  # Sets the value of L
            while j < n:
                U[k][j] = (U[i][j] / divisor) * multiplier + U[k][j]  # Sets the value of U
                j += 1
        i += 1
        k = i
    print("upper ", U)
    print("lower ", L)
    return U, L


"""
2nd step(forward substitution) of the LU Decomposition
Returns matrix D(values of each variable)
"""


def forward_substitution(L, B):
    D = {}
    n = len(L)
    i = 0
    # Creates dictionary D depending on the length of the array
    while i < n:
        D.update({"d" + str(i + 1): 0})
        i += 1

    i = 1
    j = 0
    equation = 0
    D["d1"] = B[0]
    # Solves the values of D per row
    while i < n:
        # Stores the value to variable equation until it finds a variable in D that has no value
        while j < n:
            if D["d" + str(j + 1)] == 0:
                break
            equation = equation + (L[i][j] * D["d" + str(j + 1)])
            j += 1

        # uses the sympy solve() method to solve the value of the variable that has no value
        x = Symbol('x')
        value = solve(equation + x - B[j])

        D["d" + str(i + 1)] = value[0]
        j = 0
        i += 1
        equation = 0

    # converts dictionary D to a matrix
    D_array = np.array([D["d1"]])
    i = 1
    while i < len(D):
        D_array = np.append(D_array, [D["d" + str(i + 1)]], axis=0)
        i += 1
    return D_array


def backward_substitution(U, D):
    """
    n = len(U)
    x = np.zeros_like(D)
    x[-1] = D[-1] / U[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i] = (D[i] - np.dot(U[i, i:], x[i:])) / U[i, i]
    return x
    """
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        tmp = D[i]
        for j in range(i + 1, n):
            tmp -= U[i, j] * x[j]
        x[i] = tmp / U[i, i]
    return x


#  Main method
if __name__ == "__main__":
    A, B, X = input_values()

    # for testing purposes
    # A = np.array([[1.0, -1.0, 2.0, 3.0], [2.0, 3.0, -4.0, -1.0], [-1.0, 2.0, -5.0, -4.0], [3.0, 2.0, 3.0, 7.0]])
    # B = np.array([6, -2, -7, 4])

    U, L = decomposition(A)
    D = forward_substitution(L, B)
    M = backward_substitution(U, D)
    print(" ")

    # for testing purposes
    print("D = ", D)
    print(" ")
    counter = 0
    x_value = len(M)
    while counter < x_value:
        print(X[counter], " = ", round(M[counter]))
        counter += 1
