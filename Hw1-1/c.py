import matplotlib.pyplot as plt

# Function to load numerical values from a .txt file
def load_txt(file):
    with open(file, "r") as f:
        values = f.read().split()  # Split all values on whitespace
    
    float_values = [float(value) for value in values]  # Convert each to float
    return float_values

# Function to transpose a matrix (convert rows to columns and vice versa)
def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Function to create a Vandermonde matrix for polynomial fitting
# Each row becomes: [1, x, x², ..., x^m]
def vandermonde_matrices(x, m):
    return [[x_i**j for j in range(m + 1)] for x_i in x]

# Function to multiply two matrices
def multiply_matrices(matrix1, matrix2):
    return [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*matrix2)] for X_row in matrix1]

# Optional 2x2 determinant function (not used in this version)
def determinant_two_degree(matrix, size):
    a, b = matrix[0]
    c, d = matrix[1]
    return a * d - b * c

# Function to calculate the inverse of a square matrix using Gauss-Jordan elimination
def gauss_jordan_inverse(matrix: list):
    n = len(matrix)
    
    # Create identity matrix of size n
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    
    # Augment the original matrix with the identity matrix [A | I]
    augmented = [matrix[i] + identity[i] for i in range(n)]

    for i in range(n):
        diag_element = augmented[i][i]
        if diag_element == 0:
            raise ValueError("Singular matrix, impossible reversal.")
        
        # Normalize the current row to make the diagonal 1
        augmented[i] = [x / diag_element for x in augmented[i]]
       
        # Eliminate all other entries in the current column
        for j in range(n):
            if i != j:
                factor = augmented[j][i]
                augmented[j] = [augmented[j][k] - factor * augmented[i][k] for k in range(2 * n)]

    # Extract the right-hand side of the augmented matrix — the inverse
    inverse_matrix = [row[n:] for row in augmented]
    return inverse_matrix

# Function to compute Mean Squared Error (MSE) between actual and predicted Y values
def mse(Y_true, Y_pred):
    return (1 / len(Y_true)) * sum((yt - yp) ** 2 for yt, yp in zip(Y_true, Y_pred))

# Function to predict Y values using the learned weights W
def predict(x_values, W):
    return [sum(W[i][0] * (x ** i) for i in range(len(W))) for x in x_values]

# Function to execute the polynomial fitting and plotting for a specific degree m
def start(m):
    # Load data
    x = load_txt("x.txt")
    y = load_txt("y1.txt")

    # Prepare matrices
    Y = transpose([y])                         # Turn y list into a column matrix
    X = vandermonde_matrices(x, m)             # Vandermonde matrix for x
    XT = transpose(X)                          # Transpose of X
    XTX = multiply_matrices(XT, X)             # X^T * X
    XTINV = gauss_jordan_inverse(XTX)          # Inverse of X^T * X
    XTY = multiply_matrices(XT, Y)             # X^T * Y
    W = multiply_matrices(XTINV, XTY)          # Final weights W = (X^T X)^-1 X^T Y

    # Predict and calculate error
    Y_pred = predict(x, W)
    mse_value = mse(y, Y_pred)

    # List of distinct colors for plotting weights
    colors = [
        "green", "purple", "orange", "cyan", "magenta", "yellow", "black",
        "brown", "pink", "lime", "teal", "indigo", "gold", "navy", "violet", "turquoise", "gray"
    ]

    print(f"MSE for m = {m} : {mse_value:.5f}")
    
    # Plot original dataset points
    plt.scatter(x, y, color="blue", label="Dataset points")

    # Plot the predicted polynomial curve
    plt.plot(x, Y_pred, color="red", label=f"Fitted curve (degree {m})")

    # Plot each weight value W0, W1, ..., Wm
    for i in range(len(W)):
        plt.scatter(0, W[i][0], color=colors[i % len(colors)], marker="o", label=f"W{i}")
        plt.text(0, W[i][0], f" W{i}={W[i][0]:.2f}", fontsize=12, verticalalignment='bottom', color=colors[i % len(colors)])

    # Final plot formatting
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Polynomial Curve Fitting (Degree {m})")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run curve fitting for degrees 3 and 8
def run():
    start(3)
    start(8)    

# Main execution
if __name__ == "__main__":
    run()
