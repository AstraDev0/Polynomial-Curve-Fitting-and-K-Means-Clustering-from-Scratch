import matplotlib.pyplot as plt

# === Utils ===
def load_txt(file):
    with open(file, "r") as f:
        values = f.read().splitlines()
    return [[float(num) for num in line.split()] for line in values if line.strip()]

def sqrt(x, epsilon=1e-10):
    if x < 0:
        raise ValueError("Square root of a negative number not defined for reals.")
    guess = x
    while abs(guess**2 - x) > epsilon:
        guess = (guess + x / guess) / 2
    return guess

def euclidean_distance(x_point, y_point):
    return sqrt(sum((x_i - y_i)**2 for x_i, y_i in zip(x_point, y_point)))

def allclose_manual(a, b, atol=1e-9, rtol=1e-5):
    if len(b) != len(a) and len(b[0]) != len(a[0]):
        return False
    flatten = lambda v: [item for sublist in v for item in sublist]
    for a_elem, b_elem in zip(flatten(a), flatten(b)):
        if abs(a_elem - b_elem) > atol + rtol * abs(b_elem):
            return False
    return True

def plot_clusters(clusters, distortion_values, title):
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(121, projection='3d')
    colors = ['blue', 'black', 'red', 'green', 'magenta']
    for cluster_idx, points in clusters.items():
        if points:
            x, y, z = zip(*points)
            ax.scatter(x, y, z, c=colors[cluster_idx], label=f'Cluster {cluster_idx}')
    ax.legend(); ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    ax2 = fig.add_subplot(122)
    if distortion_values:
        ax2.plot(range(len(distortion_values)), distortion_values, 'b-')
        ax2.set_title('Distortion Values Over Iterations')
        ax2.set_xlabel('Iteration'); ax2.set_ylabel('Distortion')

    plt.tight_layout()
    plt.show()

# === K-means ===
def kmeans(centroids, dataset, max_iters=100):
    k = len(centroids)
    distortion_values = []
    for iteration in range(max_iters):
        clusters = {i: [] for i in range(k)}
        total_distortion = 0
        for point in dataset:
            distances = [euclidean_distance(c, point) for c in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(point)
            total_distortion += min(distances) ** 2
            print(f"Point {point} belongs to cluster {cluster_idx}")
        distortion_values.append(total_distortion)
        plot_clusters(clusters, distortion_values, f"K-means - Iteration {iteration + 1} (After E-step)")

        new_centroids = []
        for d in range(k):
            if clusters[d]:
                new_centroids.append([sum(x) / len(clusters[d]) for x in zip(*clusters[d])])
            else:
                new_centroids.append(centroids[d])

        total_distortion = sum(
            euclidean_distance(point, new_centroids[idx]) ** 2
            for idx in range(k) for point in clusters[idx]
        )
        distortion_values.append(total_distortion)
        plot_clusters(clusters, distortion_values, f"K-means - Iteration {iteration + 1} (After M-step)")

        if allclose_manual(centroids, new_centroids):
            print(f"Converged after {iteration + 1} iterations")
            break
        centroids = new_centroids
    return centroids, distortion_values

def run():
    dataset = load_txt("Dataset_2.txt")
    centroids = [
        [0., -0.3, -2.],
        [-1.3, 1.5, 4.],
        [11.3, 3., 0.2],
        [5.7, 3.0, -2.0],
        [10., -1., 1.2]
    ]
    final_centroids, distortion_values = kmeans(centroids, dataset)
    print("Final centroids:", final_centroids)
    print("Final distortion_values:", distortion_values)

if __name__ == "__main__":
    run()
