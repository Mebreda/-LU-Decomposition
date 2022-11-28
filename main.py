from sympy.solvers import solve
from sympy import Symbol
import numpy as np
from io import StringIO
import sys
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
tmp = sys.stdout
my_result = StringIO()
sys.stdout = my_result


def input_values():
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")  # Translater to make numbers to subscripts
    degree = int(input("Enter the number of variables: "))
    X = np.array(["x1".translate(sub)])  # Array variable for X. Stores first variable and translates subscripts
    # A loop that will save unique variables depending on the input
    # Translate numbers to subscripts
    for i in range(degree - 1):
        X = np.append(X, [('x' + str(i + 2)).translate(sub)], axis=0)

    equation_cnt = 0  # Counter for the equation
    coef_cnt = 0  # Counter for the coefficients
    coefficient_array = []  # Coefficient array. Each array of values will be appended to matrix A
    B = None  # Initialization for matrix B. Set to None to be checked later if matrix is empty
    A = None  # Initialization for matrix A. Set to None to be checked later if matrix is empty
    print("Enter the coefficient of the following variables")

    # A loop that will ask for the values of the problem
    while degree > equation_cnt:
        print("Equation ", equation_cnt + 1)
        # Stores the coefficients, variables, and the value of the equation
        while degree > coef_cnt:
            coefficient = float(input(X[coef_cnt] + "= "))  # Asks for input. Reads the input as a float
            coefficient_array.append(coefficient)  # Appends coefficient to coefficient array
            coef_cnt += 1

        print("Enter the right hand side of the equation ", equation_cnt + 1)
        value = float(input("value = "))  # Asks for input. Reads the input as a float
        # Initializes matrix B if empty, otherwise append value to the matrix
        if B is None:
            B = np.array([value])
        else:
            B = np.append(B, [value], axis=0)

        # Initializes matrix A if empty, otherwise append coefficient array to the matrix
        if A is None:
            A = np.array([coefficient_array])
        else:
            A = np.append(A, [coefficient_array], axis=0)

        coefficient_array = []  # Erase values of coefficient array
        coef_cnt = 0  # Sets the coefficient counter back to 0
        equation_cnt += 1   # Counter moves to the next equation
    return A, B, X


"""
1st step(decomposition) of the LU Decomposition 
Applies the Gaussian Elimination to obtain the matrix U while simultaneously solving the values of matrix L
Returns matrix U and matrix L
"""


def decomposition(A):
    n = len(A)  # Stores length of A to n
    U = np.copy(A)  # Copies the values of A to matrix U
    L = np.zeros((n, n))  # Initializes matrix L with the same size with A. Values are set to 0

    i = 0  # First counter for the column
    j = 0  # Second counter for the column
    # Sets the middle diagonal values of the matrix L to 1
    while i < n:
        L[i][i] = 1
        i += 1

    i = 0  # Sets counter i back to 0
    k = 0  # Counter for the row
    # Gaussian Elimination for matrix U
    # Solves matrix L at the same time
    while i < n:
        # Solves the needed values per column
        while k < n - 1:
            divisor = U[i][i]  # Initializes the divisor needed for the row formula
            k += 1  # Moves to the next row
            # Condition that will check if the number is positive or negative
            # Initializes the multiplier needed for the row formula
            if U[k][i] > 0:
                multiplier = -abs(U[k][i])  # Sets the multiplier as negative
            else:
                multiplier = abs(U[k][i])  # Sets the multiplier as positive

            L[k][i] = U[k][i] / divisor  # Sets the value of L
            print()
            print("Formula for the derivation of the lower diagonal matrix:\n",U[k][i], " / ", divisor, " = ", L[k][i])  # Prints the formula for the lower diagonal matrix
            print("\nL\n", L)
            # Solves the values of the row
            while j < n:
                U[k][j] = (U[i][j] / divisor) * multiplier + U[k][j]  # Sets the value of U
                j += 1  # Moves to the next column or value

            print()
            print("U\n", U)
            print("\nRow operation: \n","( R", i + 1, " / ", divisor, ")", " * ", multiplier, " + R", k + 1)  # Prints the row formula
            j = 0
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
    D = []  # Initializes array D
    equation_str = ""  # Equation string. Used to print the equation
    print("Forward substitution")
    print("[L] [D] = [B]\n")
    # Forward Substitution
    for i in range(len(B)):
        D.append(B[i])  # Appends first value to D
        # prints the first value of D at the start of the loop
        if i == 0:
            print("d1 =", B[i])

        for j in range(i):
            equation_str = equation_str + str(L[i, j]) + "d" + str(j + 1) + " + "  # Builds the equation_str
            D[i] = D[i] - (L[i, j] * D[j])  # Solves the value of d

        D[i] = D[i] / L[i, i]  # Solves the value of d
        # Prints the equation. Avoids the first value of D
        if i != 0:
            print(equation_str + "d" + str(i + 1) + " = " + str(B[i]))
            print("d" + str(i + 1) + " =", D[i])

        equation_str = ""  # Erases the value of equation_str
    return D


"""
2nd step(forward substitution) of the LU Decomposition
Returns matrix D(values of each variable)
"""


def backward_substitution(U, D):
    # Backward Substitution for the upper triangular matrix
    print("\nBackward substitution")
    print("[U][X] = [D]")
    eq_str = " "  # Holds the equation in string form
    print(" ")  # Spacer
    u_index = U.shape[0]  # Returns an index with the dimensions of U
    x_array = np.zeros(u_index)  # Creates a new array of (u_index) which is filled with zeros

    # Condition for going through all values in the array, backwards (u_index)
    for i in range(u_index - 1, -1, -1):
        # Temporarily hold the value of D at i
        tmp = round(D[i], 14)
        # Condition for solving the values of X per row
        for j in range(i + 1, u_index):
            tmp = tmp - round(U[i, j], 14) * x_array[j]
            # Store the equation in string form
            eq_str = eq_str + "+ " + str(round(U[i, j], 14)) + "(" + "X" + str(j + 1) + ")"

        # Update the values in x_array
        x_array[i] = tmp / round(U[i, i], 14)
        # Update the string equation
        eq_str = str(tmp) + "(" + "X" + str(i + 1) + ")" + eq_str
        # Store the final equation in string form
        x_str = "X" + str(i + 1) + " = " + str(tmp) + "/" + str(round(U[i, i], 14))

        # print the equation with answer
        print(eq_str + "=", D[i])
        print(x_str + " =", x_array[i])
        eq_str = " "  # Reset the string

    return x_array


#  Main method
if __name__ == "__main__":
    # A, B, X = input_values()  # Invokes input_values() function. Returns matrices A, B, and X
    # for testing purposes
    A = np.array([[1.0, -1.0, 2.0, 3.0], [2.0, 3.0, -4.0, -1.0], [-1.0, 2.0, -5.0, -4.0], [3.0, 2.0, 3.0, 7.0]])
    B = np.array([6, -2, -7, 4])
    X = np.array(["x1", "x2", "x3", "x4"])

    U, L = decomposition(A)  # Invokes decomposition function. Returns matrices U and L
    D = forward_substitution(L, B)  # Invokes forward_substitution function. Returns array D
    M = backward_substitution(U, D)  # Invokes backward_substitution function. Returns array M
    print(" ")

    # for testing purposes
    print("D = ", D)
    print(" ")
    counter = 0  # Counter for printing the values of X
    # Prints the values of X
    while counter < len(M):
        print(X[counter], " = ", (M[counter]))
        counter += 1
