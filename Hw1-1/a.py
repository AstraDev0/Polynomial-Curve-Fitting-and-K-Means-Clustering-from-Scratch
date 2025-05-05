import matplotlib.pyplot as plt

# Function to load values from a text file and convert them to floats
def load_txt(file):
    with open(file, "r") as f:
        values = f.read().split()
    float_values = [float(value) for value in values]
    return float_values

# Function to transpose a matrix
def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Function to create a Vandermonde matrix given x values and degree m
def vandermonde_matrices(x, m):
    return [[x_i**j for j in range(m + 1)] for x_i in x]

# Function to multiply two matrices
def multiply_matrices(matrix1, matrix2):
    return [[sum(a*b for a, b in zip(X_row, Y_col)) for Y_col in zip(*matrix2)] for X_row in matrix1]

# Function to compute the determinant of a 3x3 matrix
def determinant(matrix):
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

# Function to compute predicted Y values from x and weights W
def predict(x_values, W):
    return [sum(W[i][0] * (x**i) for i in range(len(W))) for x in x_values]

# Function to compute the inverse of a 3x3 matrix using the adjugate method
def adjugate_method(matrix):
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    # Compute the adjugate matrix (cofactor matrix transposed)
    adjugate = [
        [e*i - f*h, -(b*i - c*h), b*f - c*e],
        [-(d*i - f*g), a*i - c*g, -(a*f - c*d)],
        [d*h - e*g, -(a*h - b*g), a*e - b*d]
    ]

    # Compute the determinant
    det = determinant(matrix)

    # Compute the inverse using adjugate / determinant
    inverse = [[float(adjugate[i][j] / det) for j in range(3)] for i in range(3)]
    return inverse

# Main function
def run():
    # Load x and y values from text files
    x = load_txt("x.txt")
    y = load_txt("y1.txt")
    m = 2  # Degree of the polynomial (quadratic here)

    # Format data for matrix operations
    Y = transpose([y])               # Convert y to column matrix
    X = vandermonde_matrices(x, m)   # Create Vandermonde matrix for X
    XT = transpose(X)                # Transpose of X
    XTX = multiply_matrices(XT, X)   # Compute X^T * X
    XTINV = adjugate_method(XTX)     # Inverse of X^T * X
    XTY = multiply_matrices(XT, Y)   # Compute X^T * Y
    W = multiply_matrices(XTINV, XTY)  # Final weights: W = (X^T X)^-1 X^T Y

    # Extract coefficients
    W0, W1, W2 = W[0][0], W[1][0], W[2][0]

    # Predict y values using the polynomial model
    Y_pred = predict(x, W)

    # Plot original data points
    plt.scatter(x, y, color="blue", label="Dataset points")

    # Plot the fitted polynomial curve
    plt.plot(x, Y_pred, color="red", label=f"Fitted curve (degree {m})")

    # Plot the weight values as points
    plt.scatter(0, W0, color="green", marker="o", label="W0")
    plt.scatter(1, W1, color="purple", marker="o", label="W1")
    plt.scatter(2, W2, color="orange", marker="o", label="W2")

    # Display text labels near the weights
    plt.text(0, W0, f" W0={W0:.2f}", fontsize=12, verticalalignment='bottom', color="green")
    plt.text(1, W1, f" W1={W1:.2f}", fontsize=12, verticalalignment='bottom', color="purple")
    plt.text(2, W2, f" W2={W2:.2f}", fontsize=12, verticalalignment='bottom', color="orange")

    # Plot formatting
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Polynomial Curve Fitting (Degree {m})")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# Run the main function if this script is executed
if __name__ == "__main__":
    run()
