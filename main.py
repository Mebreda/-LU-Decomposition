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
    equation_cnt = 0
    coef_cnt = 0
    coefficient_array = []
    B = None
    X = None
    A = None
    print("Enter the coefficient of the following variables")
    # A loop that will ask for the values of the problem
    while degree > equation_cnt:
        print("Equation ", equation_cnt + 1)
        # Stores the coefficients, variables, and the value of the equation
        while degree > coef_cnt:
            coefficient = float(input(variables[coef_cnt] + "= "))
            coefficient_array.append(coefficient)
            if X is None:
                # Initialize numpy array if X matrix is null
                X = np.array([variables[coef_cnt]])
            else:
                if variables[coef_cnt] not in X:
                    # Append horizontally
                    X = np.append(X, [variables[coef_cnt]], axis=0)
            coef_cnt += 1

        print("Enter the right hand side of the equation ", equation_cnt + 1)
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
        coef_cnt = 0
        equation_cnt += 1
    return A, B, X


"""
1st step(decomposition) of the LU Decomposition 
Applies the Gaussian Elimination to obtain the matrix U while simultaneously solving the values of matrix L
Returns matrix U and matrix L
"""


def decomposition(A):
    n = len(A)
    U = np.copy(A)
    L = np.zeros((n, n))

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
            print()
            print(U[k][i], " / ", divisor, " = ", L[k][i])  # Prints the formula for the lower diagonal matrix
            print("L ", L)
            while j < n:
                U[k][j] = (U[i][j] / divisor) * multiplier + U[k][j]  # Sets the value of U
                j += 1

            print()
            print("U ", U)
            print("( R", i+1, " / ", divisor, ")", " * ", multiplier, " + R", k+1)  # Prints the row formula
        i += 1
        k = i

    print()
    print("Upper diagonal matrix")
    print(U)
    print("Lower diagonal matrix")
    print(L)
    print()
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
    equation_str = ""
    D["d1"] = B[0]
    # Solves the values of D per row
    while i < n:
        # Stores the value to variable equation until it finds a variable in D that has no value
        while j < n:
            if D["d" + str(j + 1)] == 0:
                break
            equation = equation + (L[i][j] * D["d" + str(j + 1)])
            # Stores the string equation to equation_str
            equation_str = equation_str + " " + str(L[i][j]) + "d" + str(j + 1)
            equation_str = equation_str + " + d" + str(j + 2)
            j += 1
        # Prints the value of d1
        if i == 1:
            print("d1 = ", B[0])

        print(equation_str + " =", B[j])    # Prints the equation
        # uses the sympy solve() method to solve the value of the variable that has no value
        x = Symbol('x')

        value = solve(equation + x - B[j])
        D["d" + str(i + 1)] = value[0]
        print("d" + str(i + 1), " = ", value[0])    # Prints the value of d

        j = 0
        i += 1
        equation = 0
        equation_str = ""

    # converts dictionary D to a matrix
    D_array = np.array([D["d1"]])
    i = 1
    while i < len(D):
        D_array = np.append(D_array, [D["d" + str(i + 1)]], axis=0)
        i += 1
    return D_array


"""
2nd step(forward substitution) of the LU Decomposition
Returns matrix D(values of each variable)
"""


def backward_substitution(U, D):
    eq_str = " "
    print(" ")
    u_index = U.shape[0]  # Returns an index with the dimensions of U
    x_array = np.zeros(u_index)  # Creates a new array of (u_index) which is filled with zeros
    # Backward Substitution equation
    for i in range(u_index - 1, -1, -1):
        tmp = D[i]
        for j in range(i + 1, u_index):
            tmp = tmp - U[i, j] * x_array[j]
            eq_str = eq_str + " + " + str(U[i, j]) + "(" + "X" + str(j+1) + ")"
        x_array[i] = tmp / U[i, i]
        # Store the equation in a String
        eq_str = str(tmp) + "(" + "X" + str(i+1) + ")" + eq_str
        x_str = "X" + str(i+1) + " = " + str(tmp) + "/" + str(U[i, i])
        # Print the equation with answer
        print(eq_str + " =", D[i])
        print(x_str + " = ", x_array[i])
        eq_str = " "
    return x_array


#  Main method
if __name__ == "__main__":
    # A, B, X = input_values()

    # for testing purposes
    A = np.array([[1.0, -1.0, 2.0, 3.0], [2.0, 3.0, -4.0, -1.0], [-1.0, 2.0, -5.0, -4.0], [3.0, 2.0, 3.0, 7.0]])
    B = np.array([6, -2, -7, 4])
    X = np.array(["x1", "x2", "x3", "x4"])

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
        print(X[counter], " = ", (M[counter]))
        counter += 1
