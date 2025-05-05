import matplotlib.pyplot as plt

# Function to load dataset from a text file
def load_txt(file):
    with open(file, "r") as f:
        values = f.read().splitlines()  # Read lines from the file
    
    # Convert each line of values to a list of floats
    float_values = [
        [float(num) for num in line.split()] 
        for line in values if line.strip()  # Ensure empty lines are ignored
    ]
    return float_values

# Function to compute the square root using the Newton-Raphson method
def sqrt(x, epsilon=1e-10):
    if x < 0:
        raise ValueError("The square root of a negative number is not defined for the reals.")
    
    guess = x
    while abs(guess**2 - x) > epsilon:  # Iterate until desired precision
        guess = (guess + x / guess) / 2  # Update guess
    
    return guess

# Function to compute the Euclidean distance between two points
def euclidean_distance(x_point, y_point):
    return sqrt(sum((x_i - y_i)**2 for x_i, y_i in zip(x_point, y_point)))

# Function to manually check if two lists are "close enough" to each other
# Uses absolute and relative tolerance for comparison
def allclose_manual(a, b, atol=1e-9, rtol=1e-5):
    if len(b) != len(a) and len(b[0]) != len(a[0]):
        return False
    
    flatten = lambda values_list: [item for sublist in values_list for item in sublist]  # Flatten nested lists
    for a_elem, b_elem in zip(flatten(a), flatten(b)):  # Compare element by element
        diff = abs(a_elem - b_elem)
        if diff > atol + rtol * abs(b_elem):  # Check if difference exceeds tolerance
            return False
    
    return True

# Function to plot clusters and distortion values over iterations
def plot_clusters(clusters, distortion_values, title):
    fig = plt.figure(figsize=(15, 6))
    
    # Create 3D plot for cluster visualization
    ax = fig.add_subplot(121, projection='3d')
    
    colors = ['blue', 'black', 'red', 'green', 'magenta']  # Colors for clusters

    # Plot each cluster with distinct color
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
    
    # Plot distortion values (measure of how "tight" the clusters are) over iterations
    ax2 = fig.add_subplot(122)
    if len(distortion_values) > 0:
        ax2.plot(range(len(distortion_values)), distortion_values, 'b-')
        ax2.set_title('Distortion Values Over Iterations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Distortion')
    
    plt.tight_layout()
    plt.show()

# K-means clustering function
def kmeans(centroids, dataset, max_iters=100):
    k = len(centroids)  # Number of clusters
    len_dataset = len(dataset)
    clusters_list = []  # To store clusters for each iteration
    
    distortion_values = []  # To store distortion for each iteration
    
    # Perform the iterations
    for iteration in range(max_iters):
        clusters = {i: [] for i in range(k)}  # Initialize empty clusters
        total_distortion = 0  # Total distortion for this iteration
        
        # E-step: Assign each data point to the closest centroid
        for i in range(len_dataset):
            distances = [euclidean_distance(centroids[j], dataset[i]) for j in range(k)]  # Calculate distances to centroids
            min_distance = min(distances)  # Find the closest centroid
            cluster_idx = distances.index(min_distance)  # Get the index of the closest centroid
            clusters[cluster_idx].append(dataset[i])  # Assign the point to the corresponding cluster
            print(f"Point {dataset[i]} belongs to cluster {cluster_idx}")
            total_distortion += min_distance**2  # Accumulate distortion (sum of squared distances)
        
        distortion_values.append(total_distortion)  # Store the distortion value for this iteration
        clusters_list.append(clusters)  # Store clusters for this iteration
        
        # Plot clusters after the E-step (assignment of points)
        plot_clusters(clusters, distortion_values, f"K-means - Iteration {iteration + 1} (After E-step)")

        # M-step: Update the centroids by computing the mean of the points in each cluster
        new_centroids = []
        for d in range(k):
            if len(clusters[d]) > 0:
                new_centroids.append([sum(x) / len(clusters[d]) for x in zip(*clusters[d])])  # Mean of the cluster points
            else:
                new_centroids.append(centroids[d])  # If a cluster is empty, keep the old centroid
        
        # Calculating distortion after M-step (with **new** centroids)
        total_distortion = 0
        for idx in range(k):
            for point in clusters[idx]:
                total_distortion += euclidean_distance(point, new_centroids[idx]) ** 2
        distortion_values.append(total_distortion)
    
        # Plot clusters after the M-step (updating centroids)
        plot_clusters(clusters, distortion_values, f"K-means - Iteration {iteration + 1} (After M-step)")
                    
        # Check for convergence: If centroids do not change, break the loop
        if allclose_manual(centroids, new_centroids):
            print(f"Converged after {iteration + 1} iterations")
            break
    
        centroids = new_centroids  # Update centroids for the next iteration
        
    return centroids, distortion_values, clusters_list

# Function to run the K-means algorithm with an initial dataset and centroids
def run():
    dataset = load_txt("Dataset_2.txt")  # Load dataset from file
    
    # Initial centroids for the K-means algorithm
    centroids = [
        [4., -0.5, 2.],
        [2., 2.5, 1.],
        [10., 2., -1.],
        [7., 0.5, -0.5],
        [12., 1., -1.]
    ]
    
    # Perform K-means clustering
    final_centroids, distortion_values, clusters_list = kmeans(centroids, dataset)
    
    # Print the final centroids and distortion values after clustering
    print("Final centroids:", final_centroids)
    print("Final distortion values:", distortion_values)

# Main execution
if __name__ == "__main__":
    run()
