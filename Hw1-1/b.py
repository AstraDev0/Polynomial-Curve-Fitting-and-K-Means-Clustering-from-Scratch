import matplotlib.pyplot as plt

# Function to load float values from a text file
def load_txt(file):
    with open(file, "r") as f:
        values = f.read().split()  # Read all values and split by whitespace
    
    float_values = [float(value) for value in values]  # Convert to float
    return float_values

# Function to transpose a matrix
def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Function to generate a Vandermonde matrix given x values and polynomial degree m
def vandermonde_matrices(x, m):
    return [[x_i**j for j in range(m + 1)] for x_i in x]

# Function to multiply two matrices
def multiply_matrices(matrix1, matrix2):
    return [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*matrix2)] for X_row in matrix1]

# Optional (but unused): Determinant function for a 2x2 matrix (not used in this version)
def determinant_two_degree(matrix, size):
    a, b = matrix[0]
    c, d = matrix[1]
    return a * d - b * c

# Function to compute the inverse of any square matrix using Gauss-Jordan elimination
def gauss_jordan_inverse(matrix: list):
    n = len(matrix)  # Size of the matrix (assumed square)
    
    # Create identity matrix of the same size
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    
    # Create augmented matrix [A | I]
    augmented = [matrix[i] + identity[i] for i in range(n)]

    # Perform Gauss-Jordan elimination
    for i in range(n):
        diag_element = augmented[i][i]
        if diag_element == 0:
            raise ValueError("Singular matrix, impossible reversal.")  # Avoid divide-by-zero
        
        # Make the diagonal element 1
        augmented[i] = [x / diag_element for x in augmented[i]]
       
        # Make the other elements in the current column 0
        for j in range(n):
            if i != j:
                factor = augmented[j][i]
                augmented[j] = [augmented[j][k] - (factor * augmented[i][k]) for k in range(2 * n)]

    # Extract the right half of the augmented matrix (the inverse)
    inverse_matrix = [row[n:] for row in augmented]
    return inverse_matrix

# Function to predict y values using the weights W
def predict(x_values, W):
    return [sum(W[i][0] * (x**i) for i in range(len(W))) for x in x_values]

# Main function that runs the polynomial regression
def run():
    x = load_txt("x.txt")       # Load x values
    y = load_txt("y1.txt")      # Load y values
    m = 7                      # Degree of the polynomial

    Y = transpose([y])                         # Convert y to a column matrix
    X = vandermonde_matrices(x, m)             # Create Vandermonde matrix from x
    XT = transpose(X)                          # Transpose of X
    XTX = multiply_matrices(XT, X)             # X^T * X
    XTINV = gauss_jordan_inverse(XTX)          # (X^T * X)^-1
    XTY = multiply_matrices(XT, Y)             # X^T * Y
    W = multiply_matrices(XTINV, XTY)          # Final weight matrix W = (X^T X)^-1 X^T Y

    # Extract coefficients
    W0, W1, W2 = W[0][0], W[1][0], W[2][0]
    
    Y_pred = predict(x, W)                     # Compute predicted y values

    # Plot the original data points
    plt.scatter(x, y, color="blue", label="Dataset points")

    # Plot the fitted polynomial curve
    plt.plot(x, Y_pred, color="red", label=f"Fitted curve (degree {m})")

    # Plot decorations
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Polynomial Curve Fitting (Degree {m})")
    plt.legend()
    plt.grid(True)

    plt.show()

# Run the program
if __name__ == "__main__":
    run()
