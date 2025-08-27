import matplotlib.pyplot as plt

def load_txt(file):
    with open(file, "r") as f:
        return [float(v) for v in f.read().split()]

def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def vandermonde(x, m):
    return [[x_i**j for j in range(m + 1)] for x_i in x]

def multiply(A, B):
    return [[sum(a*b for a,b in zip(row, col)) for col in zip(*B)] for row in A]

def gauss_jordan_inverse(matrix):
    n = len(matrix)
    identity = [[1 if i==j else 0 for j in range(n)] for i in range(n)]
    augmented = [matrix[i] + identity[i] for i in range(n)]

    for i in range(n):
        diag = augmented[i][i]
        if diag == 0:
            raise ValueError("Singular matrix")
        augmented[i] = [x/diag for x in augmented[i]]
        for j in range(n):
            if i != j:
                factor = augmented[j][i]
                augmented[j] = [augmented[j][k] - factor*augmented[i][k] for k in range(2*n)]
    return [row[n:] for row in augmented]

def predict(x_values, W):
    return [sum(W[i][0]*(x**i) for i in range(len(W))) for x in x_values]

def run():
    x = load_txt("x.txt")
    y = load_txt("y1.txt")
    m = 7

    Y = transpose([y])
    X = vandermonde(x, m)
    XT = transpose(X)
    W = multiply(gauss_jordan_inverse(multiply(XT,X)), multiply(XT,Y))

    Y_pred = predict(x, W)

    plt.scatter(x, y, color="blue", label="Data points")
    plt.plot(x, Y_pred, color="red", label=f"Fitted curve (degree {m})")
    plt.xlabel("X"); plt.ylabel("Y"); plt.title(f"Polynomial Curve Fitting (Degree {m})")
    plt.legend(); plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run()
