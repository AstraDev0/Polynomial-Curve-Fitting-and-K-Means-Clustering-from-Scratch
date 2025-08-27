import matplotlib.pyplot as plt

def load_txt(file):
    with open(file, "r") as f:
        values = f.read().split()
    return [float(v) for v in values]

def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def vandermonde(x, m):
    return [[x_i**j for j in range(m + 1)] for x_i in x]

def multiply(A, B):
    return [[sum(a*b for a,b in zip(row, col)) for col in zip(*B)] for row in A]

def determinant(M):
    a,b,c = M[0]; d,e,f = M[1]; g,h,i = M[2]
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

def inverse_3x3(M):
    a,b,c = M[0]; d,e,f = M[1]; g,h,i = M[2]
    adj = [[e*i - f*h, -(b*i - c*h), b*f - c*e],
           [-(d*i - f*g), a*i - c*g, -(a*f - c*d)],
           [d*h - e*g, -(a*h - b*g), a*e - b*d]]
    det = determinant(M)
    return [[adj[i][j]/det for j in range(3)] for i in range(3)]

def predict(x, W):
    return [sum(W[i][0]*x**i for i in range(len(W))) for x in x]

def run():
    x = load_txt("x.txt")
    y = load_txt("y1.txt")
    m = 2

    Y = transpose([y])
    X = vandermonde(x, m)
    XT = transpose(X)
    W = multiply(inverse_3x3(multiply(XT,X)), multiply(XT,Y))
    Y_pred = predict(x, W)

    plt.scatter(x, y, color="blue", label="Data")
    plt.plot(x, Y_pred, color="red", label=f"Degree {m}")
    for i, w in enumerate(W):
        plt.scatter(i, w[0], label=f"W{i}")
        plt.text(i, w[0], f"{w[0]:.2f}", fontsize=12, color="black", verticalalignment='bottom')
    plt.xlabel("X"); plt.ylabel("Y"); plt.title(f"Polynomial Curve Fitting (Degree {m})")
    plt.legend(); plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run()
