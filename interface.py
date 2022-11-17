import tkinter
from tkinter import Tk, Label, StringVar, Button, Entry, simpledialog
from PIL import Image, ImageTk
from functools import partial

import numpy as np


# callback function to get your StringVars
def get_mat(matrix_size, text_var, A, B, X):
    # Create a matrix of N x N + 1, where the 1 increment
    # is allocated for the right hand side
    coef_arr = np.zeros((matrix_size, matrix_size + 1), dtype=float)
    # give the temp variable a top-level function scope
    coef = 0.0
    for i in range(matrix_size):
        # Represents 1 row in a 2D array
        tempArr = []
        for j in range(matrix_size + 1):
            tempArr.append(float(text_var[i][j].get()))
        coef_arr[i] = tempArr.copy()
        # Exclude the RHS
        A[i] = tempArr.copy()[:len(tempArr) - 1]
        B.append(float(tempArr[len(tempArr)-1]))
    # Generate X Matrix
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    for i in range(matrix_size):
        X.append(('x' + str(i + 1)).translate(sub))
    print("A: ", A, sep='\n')
    print("B: ", B, sep='\n')
    print("X: ", X, sep='\n')

def generate_interface(root, matrix_size, A, B, X):
    text_var = []
    entries = []

    x2 = 0
    y2 = 0

    rows, cols = matrix_size, (matrix_size + 1)

    for i in range(rows):
        # append an empty list to your two arrays
        # so you can append to those later
        text_var.append([])
        entries.append([])
        for j in range(cols):
            # append your StringVar and Entry
            text_var[i].append(StringVar())
            entries[i].append(Entry(root, textvariable=text_var[i][j], width=3))
            entries[i][j].place(x=60 + x2, y=50 + y2)
            x2 += 30
        y2 += 30
        x2 = 0
    button = Button(root, text="Submit", width=15, command=partial(get_mat, matrix_size, text_var, A, B, X))
    button.place(x=160, y=140)
    showButtton = Button(root, text="Submit", width=15, command=partial(show_matrix, root, A))
    showButtton.place(x=260, y=140)
    return text_var

def show_matrix(root, arr):
    print(arr)
    rows = len(arr)
    cols = len(arr[0])
    # Window logic
    win = tkinter.Toplevel(root)
    # Receive events and prevent users from interacting with the root window
    win.grab_set()
    canvas = tkinter.Canvas(root, height = 300, width=500)
    canvas.pack()
    # Display data logic
    for c in range(cols):
        label = tkinter.Label(win, text=str(c))
        label.grid(row=0, column=c+1)
    all_enties = []
    for r in range(rows):
        entries_row = []
        label = tkinter.Label(win, text=str(r+1))
        label.grid(row=r+1, column=0)
        for c in range(cols):
            e = tkinter.Entry(win, width=5)
            e.insert('end', arr[r][c])
            e.grid(row=r+1, column=c+1)
            entries_row.append(e)
        all_enties.append(entries_row)





if __name__ == '__main__':

    root = Tk()
    root.title("Matrix")
    root.geometry("650x500+120+120")
    root.resizable(False, False)

    matrix_size = simpledialog.askinteger("Input", "Number of variables?", parent=root)

    # Initialize the A matrix
    # The A matrix consists of the coefficients of the left hand side,
    # Which contains the system of equations
    A = np.zeros((matrix_size, matrix_size), dtype=float)
    # Right hand side of the equations
    B = []
    # Matrix to hold X variables
    X = []

    text_var = generate_interface(root, matrix_size, A, B, X)

    # get_mat(matrix_size, text_var)
    root.mainloop()