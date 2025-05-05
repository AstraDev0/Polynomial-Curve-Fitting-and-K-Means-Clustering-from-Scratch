import matplotlib.pyplot as plt
import random

# Function to load dataset from a text file and convert it to a list of float values
def load_txt(file):
    with open(file, "r") as f:
        values = f.read().splitlines()  # Read all lines from the file
    # Convert each line into a list of floats and return as a list of points
    float_values = [
        [float(num) for num in line.split()] 
        for line in values if line.strip()  # Skip empty lines
    ]
    return float_values

# Function to calculate square root using Newton's method
def sqrt(x, epsilon=1e-10):
    if x < 0:
        raise ValueError("The square root of a negative number is not defined for the reals.")
    
    guess = x
    # Iteratively improve guess until the difference between guess^2 and x is small enough
    while abs(guess**2 - x) > epsilon:
        guess = (guess + x / guess) / 2
    
    return guess

# Function to calculate Euclidean distance between two points
def euclidean_distance(x_point, y_point):
    return sqrt(sum((x_i - y_i)**2 for x_i, y_i in zip(x_point, y_point)))

# Function to compute total distortion (sum of squared distances from points to the center)
def compute_distortion(points, center):
    return sum(euclidean_distance(point, center)**2 for point in points)

# Function to check if two centroid lists are close enough to each other (convergence check)
def allclose_manual(a, b, atol=1e-9, rtol=1e-5):
    if len(b) != len(a) and len(b[0]) != len(a[0]):
        return False
    
    # Flatten both lists of points and compare each pair
    flatten = lambda values_list: [item for sublist in values_list for item in sublist]
    for a_elem, b_elem in zip(flatten(a), flatten(b)):
        diff = abs(a_elem - b_elem)
        if diff > atol + rtol * abs(b_elem):
            return False
    
    return True

# Function to compute the mean of a list of points (centroid calculation)
def mean(points):
    return [sum(x) / len(points) for x in zip(*points)]

# K-means clustering algorithm
def kmeans(centroids, dataset, max_iters=100):
    len_dataset = len(dataset)
    clusters_list = []
    k = len(centroids)
    
    distortion_values = []  # To track the distortion at each iteration
    
    for iteration in range(max_iters):
        # Initialize empty clusters for each centroid
        clusters = {i: [] for i in range(k)}
        total_distortion = 0  # To accumulate the total distortion for the current iteration
        
        # E-step: Assign each point to the closest centroid
        for i in range(len_dataset):
            distances = [euclidean_distance(centroids[j], dataset[i]) for j in range(k)]
            min_distance = min(distances)
            cluster_idx = distances.index(min_distance)
            clusters[cluster_idx].append(dataset[i])
            print(f"Point {dataset[i]} belongs to cluster {cluster_idx}")
            total_distortion += min_distance**2
        
        distortion_values.append(total_distortion)  # Store distortion for the current iteration
        clusters_list.append(clusters)  # Store the clusters for plotting
        
        # M-step: Recompute centroids as the mean of the points in each cluster
        new_centroids = []
        for d in range(k):
            if len(clusters[d]) > 0:
                new_centroids.append([sum(x) / len(clusters[d]) for x in zip(*clusters[d])])
            else:
                new_centroids.append(centroids[d])  # Handle empty clusters by keeping the old centroid

        # Calculating distortion after M-step (with **new** centroids)
        total_distortion = 0
        for idx in range(k):
            for point in clusters[idx]:
                total_distortion += euclidean_distance(point, new_centroids[idx]) ** 2
        distortion_values.append(total_distortion)
        
        # Plot the clusters after M-step
        plot_clusters(list(clusters.values()), f"K-means - Iteration {iteration + 1}")
        
        # Check for convergence (if centroids do not change significantly)
        if allclose_manual(centroids, new_centroids):
            print(f"Converged after {iteration + 1} iterations")
            break
    
        centroids = new_centroids  # Update centroids for the next iteration
        
    return centroids, distortion_values, clusters_list

# Function to perform a binary split on a cluster of points (used in non-uniform binary split)
def split_class(points, center):
    if not points:
        return [], []  # Return empty clusters if there are no points
    
    d = len(center)  # Dimension of the points (3D in this case)
    # Add small random offsets to the center to create two new "centers"
    v = [random.uniform(-0.01, 0.01) for _ in range(d)]
    
    center_plus = [center[i] + v[i] for i in range(d)]
    center_minus = [center[i] - v[i] for i in range(d)]
    
    cluster_a, cluster_b = [], []
    
    # Split the points into two clusters based on their distance to the new centers
    for p in points:
        if euclidean_distance(p, center_plus) < euclidean_distance(p, center_minus):
            cluster_a.append(p)
        else:
            cluster_b.append(p)

    return cluster_a, cluster_b

# Function to visualize the clusters in 3D
def plot_clusters(clusters, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['blue', 'black', 'red', "green", 'magenta']
    
    # Plot each cluster with a different color
    for i, cluster in enumerate(clusters):
        if cluster:
            x = [point[0] for point in cluster]
            y = [point[1] for point in cluster]
            z = [point[2] for point in cluster]
            ax.scatter(x, y, z, color=colors[i % len(colors)], label=f'Cluster {i}')
    
    ax.set_title(title)
    ax.legend()
    plt.show()

# Function for non-uniform binary split to create multiple clusters from a dataset
def non_uniform_binary_split(dataset, initial_centroids):
    clusters = [dataset]
    centers = [mean(dataset)]
    i = 0
    
    # Keep splitting the dataset until the desired number of clusters is reached
    while len(clusters) < len(initial_centroids):
        distortions = [compute_distortion(cluster, mean(cluster)) for cluster in clusters]
        argmax = distortions.index(max(distortions))  # Find the cluster with the maximum distortion
        
        cluster_to_split = clusters[argmax]
        center_to_split = centers[argmax]
        
        clusters.pop(argmax)  # Remove the selected cluster from the list
        centers.pop(argmax)  # Remove the corresponding centroid
        
        Xa, Xb = split_class(cluster_to_split, center_to_split)  # Split the cluster into two
        
        # Add the new clusters and centroids to the list
        if Xa and Xb:
            clusters.extend([Xa, Xb])
            centers.extend([mean(Xa), mean(Xb)])
            
        # Plot the clusters at each iteration
        plot_clusters(clusters, f"Non-uniform binary split - Iteration {i}")
        i = i + 1
        
    return centers, clusters

# Main function to run both K-means and non-uniform binary split
def run():
    dataset = load_txt("Dataset_2.txt")  # Load dataset from file
    
    # Initial centroids for the K-means algorithm
    centroids = [[0., -0.3, -2.],
                [-1.3, 1.5, 4.],
                [11.3, 3., 0.2],
                [5.7, 3.0, -2.0],
                [10., -1., 1.2]]
    
    # Run non-uniform binary split to create initial clusters
    non_uniform_binary_split(dataset, centroids)
    
    # Run K-means clustering
    kmeans(centroids, dataset)

# Entry point to execute the program
if __name__ == "__main__":
    run()
