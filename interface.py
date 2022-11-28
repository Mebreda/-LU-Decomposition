from tkinter import *
from tkinter import ttk, simpledialog
from functools import partial
from main import *


def generate_matrix_entries(frame, matrix_size):
    text_var = []
    entries = []

    rows, cols = matrix_size, (matrix_size + 1)

    for row in range(rows):
        # temporary 1D array for the 2D array (matrix)
        text_var.append([])
        entries.append([])
        for col in range(cols):
            text_var[row].append(StringVar())
            entries[row].append(Entry(frame, textvariable=text_var[row][col], width=6))
            entries[row][col].grid(row=row, column=col)
    return text_var, entries


def generate_equation_frame(frame, mat_size):
    equation_box = Text(frame, height=mat_size + 2, width=mat_size * 16)
    equation_box.grid(row=0, column=0)


def generate_diagonal(frame, mat_size):
    diagonal_matrix_box = Text(frame, height=mat_size + 2, width=mat_size * 16)
    diagonal_matrix_box.grid(row=0, column=0)


def generate_solution(frame, mat_size):
    sol_frm = ttk.Frame(frame, padding="12 10 12 12")
    deriv_frm = ttk.LabelFrame(frame, padding="10 0 10 10")
    sol_box = Text(sol_frm, height=mat_size + 2, width=mat_size * 16)
    deriv_btn = ttk.Button(deriv_frm, text="Show Derivation", command=partial(show_derivation, frame))
    sol_box.grid(row=0, column=0)
    deriv_btn.grid(row=10, column=0)

    sol_frm.grid(row=0, column=0)
    deriv_frm.grid(row=1, column=0)




def generate_controls(frame, text_var, entries_vars, eqn_widget, right_frame, A, B, X):
    # text_var, eqn_widget, A, B, X
    solvebtn = ttk.Button(frame, text="Solve",
                          command=partial(solve_handler, text_var, eqn_widget, right_frame, A, B, X))
    solvebtn.grid(row=0, column=0)
    clearbtn = ttk.Button(frame, text="Clear", command=partial(clear_fields, entries_vars, eqn_widget))
    clearbtn.grid(row=0, column=1)


def extract_matrix_input(text_var, A, B, X):
    # Create a matrix of N X (N + 1), where the 1 increment
    # is allocated for the RHS (matrix B)
    coef_arr = np.zeros((len(text_var), (len(text_var) + 1)), dtype=float)

    for row in range(len(text_var)):
        # Represents 1 row in a 2D array
        tempArr = []
        for col in range(len(text_var[0])):
            tempArr.append(float(text_var[row][col].get()))
        coef_arr[row] = tempArr.copy()
        # Exclude the RHS
        A[row] = tempArr.copy()[:len(tempArr) - 1]
        B.append(float(tempArr[len(tempArr) - 1]))
    # Generate X Matrix
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    for i in range(len(text_var)):
        X.append(('x' + str(i + 1)).translate(sub))


def show_system(eqn_widget, A, B, X):
    sys_eqn_str = ""
    operand = ""
    for row_index, row in enumerate(A):
        for element_index, element in enumerate(row):
            temp_str = str(abs(element)) + X[element_index]
            if element_index + 1 != len(row):
                next = A[row_index][element_index + 1]
                if next > 0:
                    operand = " + "
                    temp_str += operand
                elif next < 0:
                    operand = " - "
                    temp_str += operand
            sys_eqn_str += temp_str
        sys_eqn_str += (" = " + str(B[row_index]) + '\n')
    eqn_widget.delete('1.0', 'end')
    eqn_widget.insert('1.0', sys_eqn_str)


def clear_mat(mat):
    for row_index, row in enumerate(mat):
        for e_index, e in enumerate(row):
            mat[row_index][e_index] = 0


def clear_arr(arr):
    for i, e in enumerate(arr):
        if isinstance(e, str):
            arr[i] = 0
        else:
            arr[i] = 0


def clear_fields(entries_vars, eqn_widget):
    blank = ""

    for row in range(len(entries_vars)):
        # Represents 1 row in a 2D array
        for col in range(len(entries_vars[0])):
            entries_vars[row][col].delete(0, END)

    eqn_widget.delete('1.0', 'end')
    update_right_subframe(right_frame.children['!labelframe'], blank)
    update_right_subframe(right_frame.children['!labelframe2'], blank)
    update_d_subframe(right_frame.children['!labelframe3'], blank)

    update_sol_subframe(right_frame.children['!labelframe4'].children['!frame'], blank)


def solve_handler(text_var, eqn_widget, right_frame, A, B, X):

    extract_matrix_input(text_var, A, B, X)

    show_system(eqn_widget, A, B, X)

    U, L = decomposition(A.copy())
    D = forward_substitution(L, B.copy())
    sol = backward_substitution(U, D.copy())

    update_right_subframe(right_frame.children['!labelframe'], U)
    update_right_subframe(right_frame.children['!labelframe2'], L)
    update_d_subframe(right_frame.children['!labelframe3'], D)

    update_sol_subframe(right_frame.children['!labelframe4'].children['!frame'], sol)


def update_right_subframe(frame, mat):
    frame.children['!text'].delete('1.0', 'end')
    frame.children['!text'].insert('1.0', str(mat))


def update_d_subframe(frame, D):
    frame_str = ""
    tr_table = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    substr = []
    for i in range(len(D)):
        substr.append(('d' + str(i + 1)).translate(tr_table))

    for i, d in enumerate(D):
        frame_str += substr[i] + " = " + str(d) + '\n'

    frame.children['!text'].delete('1.0', 'end')
    frame.children['!text'].insert('1.0', frame_str)


def update_sol_subframe(frame, sol):
    frame_str = ""
    tr_table = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    substr = []
    for i in range(len(sol)):
        substr.append(('x' + str(i + 1)).translate(tr_table))

    for i, d in enumerate(sol):
        frame_str += substr[i] + " = " + str(d) + '\n'

    frame.children['!text'].delete('1.0', 'end')
    frame.children['!text'].insert('1.0', frame_str)


def show_derivation(root):
    window = Toplevel(root)
    s = my_result.getvalue()


    window_frame = ttk.Frame(window, padding="12 10 12 12")
    window_frame.grid(row=0, column=0)
    
    deriv_box = Text(window_frame, height=16, width=80)
    deriv_box.grid(row=0, column=0)

    # Scrollbar logic
    scrollbar = ttk.Scrollbar(window_frame, command=deriv_box.yview)
    scrollbar.grid(row=0, column=1, sticky='nsew')
    deriv_box['yscrollcommand'] = scrollbar.set
    deriv_box.delete('1.0', 'end')
    deriv_box.insert('1.0', s)
    
    window.title("Derivation Window")
    # Label(window, text=s, font=('Mistral 18 bold')).place(x=150, y=80)



if __name__ == '__main__':
    # Create the root frame
    root = Tk()
    root.title("LU Decomposition Utility")
    # Prompt number of variables
    matrix_size = simpledialog.askinteger("Input", "Number of variables?", parent=root)
    # Initialize the A matrix
    # The A matrix consists of the coefficients of the left hand side,
    # Which contains the system of equations
    A = np.zeros((matrix_size, matrix_size), dtype=float)
    # Right hand side of the equations
    B = []
    # Matrix to hold X variables
    X = []
    # Create the content frame (main frame)
    main_frame = ttk.Frame(root, padding="12 12 12 12")
    # Standard practice, a top level frame must hold other components
    main_frame.grid(column=0, row=0, sticky=(N, W, E, S))
    # Hold the matrix input fields and the text field that shows
    # the inputted equation
    left_frame = ttk.LabelFrame(main_frame, padding="12 10 12 12")
    left_frame.grid(row=0, column=0)
    # Populate the left frame
    matrix_frame = ttk.LabelFrame(left_frame, padding="12 10 12 12", text="Matrix")
    equation_frame = ttk.LabelFrame(left_frame, padding="12 10 12 12", text="System of Equations")
    control_frame = ttk.LabelFrame(left_frame, padding="12 10 12 12", text="Controls")

    matrix_frame.grid(row=1, column=0)
    equation_frame.grid(row=2, column=0)
    control_frame.grid(row=3, column=0)

    text_vars, entries_vars = generate_matrix_entries(matrix_frame, matrix_size)
    generate_equation_frame(equation_frame, matrix_size)

    # Right frame will contain the L, U and the solutions to the equation
    right_frame = ttk.LabelFrame(main_frame, padding="12 12 12 12")
    right_frame.grid(row=0, column=1)

    generate_controls(control_frame, text_vars, entries_vars, equation_frame.children['!text'], right_frame, A, B, X)

    # Populate the right frame
    # Upper Diagonal Frame
    upper_diag_frame = ttk.LabelFrame(right_frame, padding="12 10 12 12", text="Upper Diagonal Matrix")
    upper_diag_frame.grid(row=0, column=0)
    generate_diagonal(upper_diag_frame, matrix_size)
    # Lower Diagonal Frame
    lower_diag_frame = ttk.LabelFrame(right_frame, padding="12 10 12 12", text="Lower Diagonal Matrix")
    lower_diag_frame.grid(row=1, column=0)
    generate_diagonal(lower_diag_frame, matrix_size)
    # D-Matrix Frame
    d_matrix_frame = ttk.LabelFrame(right_frame, padding="12 10 12 12", text="D-Matrix")
    d_matrix_frame.grid(row=2, column=0)
    generate_diagonal(d_matrix_frame, matrix_size)

    # Solution to the problem
    solution = ttk.LabelFrame(right_frame, padding="12 10 12 12", text="Solution to the system")
    solution.grid(row=3, column=0)
    generate_solution(solution, matrix_size)

    root.mainloop()
