import matplotlib.pyplot as plt

# Utility functions (loading data, math operations, plotting)
def load_txt(file):
    with open(file, "r") as f:
        values = f.read().splitlines()
    float_values = [
        [float(num) for num in line.split()] 
        for line in values if line.strip()
    ]
    return float_values

def sqrt(x, epsilon=1e-10):
    if x < 0:
        raise ValueError("Square root of negative number is not defined.")
    guess = x
    while abs(guess**2 - x) > epsilon:
        guess = (guess + x / guess) / 2
    return guess

def euclidean_distance(x_point, y_point):
    return sqrt(sum((x_i - y_i)**2 for x_i, y_i in zip(x_point, y_point)))

def allclose_manual(a, b, atol=1e-9, rtol=1e-5):
    if len(b) != len(a) and len(b[0]) != len(a[0]):
        return False
    flatten = lambda values_list: [item for sublist in values_list for item in sublist]
    for a_elem, b_elem in zip(flatten(a), flatten(b)):
        diff = abs(a_elem - b_elem)
        if diff > atol + rtol * abs(b_elem):
            return False
    return True

def plot_clusters(clusters, distortion_values, title):
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(121, projection='3d')
    colors = ['blue', 'black', 'red', 'green', 'magenta']
    for cluster_idx, points in clusters.items():
        if points:
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            z_coords = [point[2] for point in points]
            ax.scatter(x_coords, y_coords, z_coords, c=colors[cluster_idx], label=f'Cluster {cluster_idx}')
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax2 = fig.add_subplot(122)
    if len(distortion_values) > 0:
        ax2.plot(range(len(distortion_values)), distortion_values, 'b-')
        ax2.set_title('Distortion Values Over Iterations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Distortion')
    plt.tight_layout()
    plt.show()

# K-means clustering algorithm
def kmeans(centroids, dataset, max_iters=100):
    k = len(centroids)
    len_dataset = len(dataset)
    clusters_list = []
    distortion_values = []
    for iteration in range(max_iters):
        clusters = {i: [] for i in range(k)}
        total_distortion = 0
        for i in range(len_dataset):
            distances = [euclidean_distance(centroids[j], dataset[i]) for j in range(k)]
            min_distance = min(distances)
            cluster_idx = distances.index(min_distance)
            clusters[cluster_idx].append(dataset[i])
            print(f"Point {dataset[i]} belongs to cluster {cluster_idx}")
            total_distortion += min_distance**2
        distortion_values.append(total_distortion)
        clusters_list.append(clusters)
        plot_clusters(clusters, distortion_values, f"K-means - Iteration {iteration + 1} (After E-step)")
        new_centroids = []
        for d in range(k):
            if len(clusters[d]) > 0:
                new_centroids.append([sum(x) / len(clusters[d]) for x in zip(*clusters[d])])
            else:
                new_centroids.append(centroids[d])
        total_distortion = 0
        for idx in range(k):
            for point in clusters[idx]:
                total_distortion += euclidean_distance(point, new_centroids[idx]) ** 2
        distortion_values.append(total_distortion)
        plot_clusters(clusters, distortion_values, f"K-means - Iteration {iteration + 1} (After M-step)")
        if allclose_manual(centroids, new_centroids):
            print(f"Converged after {iteration + 1} iterations")
            break
        centroids = new_centroids
    return centroids, distortion_values, clusters_list

def run():
    dataset = load_txt("Dataset_2.txt")
    centroids = [
        [4., -0.5, 2.],
        [2., 2.5, 1.],
        [10., 2., -1.],
        [7., 0.5, -0.5],
        [12., 1., -1.]
    ]
    final_centroids, distortion_values, clusters_list = kmeans(centroids, dataset)
    print("Final centroids:", final_centroids)
    print("Final distortion values:", distortion_values)

if __name__ == "__main__":
    run()
