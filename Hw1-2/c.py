import matplotlib.pyplot as plt
import random

# --- Utils ---
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

def euclidean_distance(a, b):
    return sqrt(sum((x - y)**2 for x, y in zip(a, b)))

def compute_distortion(points, center):
    return sum(euclidean_distance(p, center)**2 for p in points)

def allclose_manual(a, b, atol=1e-9, rtol=1e-5):
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return False
    flatten = lambda L: [x for sub in L for x in sub]
    return all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(flatten(a), flatten(b)))

def mean(points):
    return [sum(x) / len(points) for x in zip(*points)]

# --- K-means ---
def kmeans(centroids, dataset, max_iters=100):
    k = len(centroids)
    distortion_values, clusters_list = [], []

    for iteration in range(max_iters):
        clusters = {i: [] for i in range(k)}
        total_distortion = 0

        # E-step
        for point in dataset:
            distances = [euclidean_distance(c, point) for c in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(point)
            total_distortion += min(distances)**2
            print(f"Point {point} -> Cluster {cluster_idx}")

        distortion_values.append(total_distortion)
        clusters_list.append(clusters)

        # M-step
        new_centroids = []
        for d in range(k):
            new_centroids.append(mean(clusters[d]) if clusters[d] else centroids[d])

        total_distortion = sum(
            euclidean_distance(p, new_centroids[idx])**2
            for idx in range(k) for p in clusters[idx]
        )
        distortion_values.append(total_distortion)

        plot_clusters(list(clusters.values()), f"K-means - Iteration {iteration + 1}")

        if allclose_manual(centroids, new_centroids):
            print(f"Converged after {iteration + 1} iterations")
            break

        centroids = new_centroids

    return centroids, distortion_values, clusters_list

# --- Binary Split ---
def split_class(points, center):
    if not points:
        return [], []
    d = len(center)
    v = [random.uniform(-0.01, 0.01) for _ in range(d)]
    c_plus = [center[i] + v[i] for i in range(d)]
    c_minus = [center[i] - v[i] for i in range(d)]
    a, b = [], []
    for p in points:
        (a if euclidean_distance(p, c_plus) < euclidean_distance(p, c_minus) else b).append(p)
    return a, b

def non_uniform_binary_split(dataset, initial_centroids):
    clusters, centers, i = [dataset], [mean(dataset)], 0
    while len(clusters) < len(initial_centroids):
        distortions = [compute_distortion(c, mean(c)) for c in clusters]
        idx = distortions.index(max(distortions))
        cluster, center = clusters.pop(idx), centers.pop(idx)
        Xa, Xb = split_class(cluster, center)
        if Xa and Xb:
            clusters.extend([Xa, Xb])
            centers.extend([mean(Xa), mean(Xb)])
        plot_clusters(clusters, f"Non-uniform binary split - Iteration {i}")
        i += 1
    return centers, clusters

# --- Plotting ---
def plot_clusters(clusters, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['blue', 'black', 'red', "green", 'magenta']
    for i, cluster in enumerate(clusters):
        if cluster:
            x, y, z = zip(*cluster)
            ax.scatter(x, y, z, color=colors[i % len(colors)], label=f'Cluster {i}')
    ax.set_title(title)
    ax.legend()
    plt.show()

# --- Main ---
def run():
    dataset = load_txt("Dataset_2.txt")
    centroids = [
        [0., -0.3, -2.],
        [-1.3, 1.5, 4.],
        [11.3, 3., 0.2],
        [5.7, 3.0, -2.0],
        [10., -1., 1.2]
    ]
    non_uniform_binary_split(dataset, centroids)
    kmeans(centroids, dataset)

if __name__ == "__main__":
    run()
