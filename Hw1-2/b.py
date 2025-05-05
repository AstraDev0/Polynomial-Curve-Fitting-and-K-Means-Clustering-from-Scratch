import matplotlib.pyplot as plt

# Function to load the dataset from a text file
def load_txt(file):
    # Open the file and read all lines
    with open(file, "r") as f:
        values = f.read().splitlines()
    
    # Convert each line into a list of floats and return the 2D list of points
    float_values = [
        [float(num) for num in line.split()] 
        for line in values if line.strip()  # Skip empty lines
    ]
    return float_values

# Function to calculate the square root of a number using Newton's method
def sqrt(x, epsilon=1e-10):
    # Raise an error for negative inputs as square root is not defined for them in real numbers
    if x < 0:
        raise ValueError("The square root of a negative number is not defined for the reals.")
    
    guess = x
    # Continue improving the guess until the difference between guess^2 and x is smaller than epsilon
    while abs(guess**2 - x) > epsilon:
        guess = (guess + x / guess) / 2
    
    return guess

# Function to compute the Euclidean distance between two points in 3D space
def euclidean_distance(x_point, y_point):
    # Sum of squared differences between the coordinates and take the square root
    return sqrt(sum((x_i - y_i)**2 for x_i, y_i in zip(x_point, y_point)))

# Function to check if two lists of points (centroids) are close enough to each other
def allclose_manual(a, b, atol=1e-9, rtol=1e-5):
    # Ensure both lists have the same length and structure
    if len(b) != len(a) and len(b[0]) != len(a[0]):
        return False
    
    # Flatten the lists and compare corresponding elements
    flatten = lambda values_list: [item for sublist in values_list for item in sublist]
    for a_elem, b_elem in zip(flatten(a), flatten(b)):
        diff = abs(a_elem - b_elem)
        # Check if the difference exceeds the absolute tolerance or relative tolerance
        if diff > atol + rtol * abs(b_elem):
            return False
    
    return True

# Function to plot the clusters and distortion values over iterations
def plot_clusters(clusters, distortion_values, title):
    # Create a figure with two subplots: 3D scatter plot and distortion plot
    fig = plt.figure(figsize=(15, 6))
    
    # 3D scatter plot for the clusters
    ax = fig.add_subplot(121, projection='3d')
    colors = ['blue', 'black', 'red', 'green', 'magenta']

    # Loop through each cluster and plot the points in 3D space
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
    
    # Plot distortion values over iterations
    ax2 = fig.add_subplot(122)
    if len(distortion_values) > 0:
        ax2.plot(range(len(distortion_values)), distortion_values, 'b-')
        ax2.set_title('Distortion Values Over Iterations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Distortion')
    
    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

# Main K-means clustering function
def kmeans(centroids, dataset, max_iters=100):
    k = len(centroids)  # Number of clusters
    len_dataset = len(dataset)  # Number of data points
    
    distortion_values = []  # To track the total distortion at each iteration
    
    # Perform the algorithm for a maximum of max_iters iterations
    for iteration in range(max_iters):
        # Create empty clusters for each centroid
        clusters = {i: [] for i in range(k)}
        total_distortion = 0  # Initialize total distortion
        
        # E-step: Assign each point to the nearest centroid
        for i in range(len_dataset):
            distances = [euclidean_distance(centroids[j], dataset[i]) for j in range(k)]  # Calculate distances to all centroids
            min_distance = min(distances)  # Find the minimum distance
            cluster_idx = distances.index(min_distance)  # Find the index of the closest centroid
            clusters[cluster_idx].append(dataset[i])  # Assign the point to the corresponding cluster
            print(f"Point {dataset[i]} belongs to cluster {cluster_idx}")
            total_distortion += min_distance**2  # Add the squared distance to total distortion
        
        # Append the distortion value for the current iteration
        distortion_values.append(total_distortion)
        
        # Plot the clusters and distortion after the E-step
        plot_clusters(clusters, distortion_values, f"K-means - Iteration {iteration + 1} (After E-step)")

        # M-step: Recompute centroids as the mean of the points in each cluster
        new_centroids = []
        for d in range(k):
            if len(clusters[d]) > 0:
                # Calculate the new centroid as the mean of the points in the cluster
                new_centroids.append([sum(x) / len(clusters[d]) for x in zip(*clusters[d])])
            else:
                # If the cluster is empty, keep the old centroid (or reinitialize randomly)
                new_centroids.append(centroids[d])
                
        # Calculating distortion after M-step (with **new** centroids)
        total_distortion = 0
        for idx in range(k):
            for point in clusters[idx]:
                total_distortion += euclidean_distance(point, new_centroids[idx]) ** 2
        distortion_values.append(total_distortion)
        
        # Plot the clusters and distortion after the M-step
        plot_clusters(clusters, distortion_values, f"K-means - Iteration {iteration + 1} (After M-step)")

        # Check for convergence: if the centroids haven't changed, stop the algorithm
        if allclose_manual(centroids, new_centroids):
            print(f"Converged after {iteration + 1} iterations")
            break
    
        # Update centroids for the next iteration
        centroids = new_centroids
        
    return centroids, distortion_values  # Return final centroids and distortion values

# Function to run the K-means algorithm
def run():
    # Load the dataset from a text file
    dataset = load_txt("Dataset_2.txt")
    
    # Initial centroids for the K-means algorithm (can be chosen randomly or manually)
    centroids = [[0., -0.3, -2.],
                [-1.3, 1.5, 4.],
                [11.3, 3., 0.2],
                [5.7, 3.0, -2.0],
                [10., -1., 1.2]]
    
    # Run the K-means algorithm
    final_centroids, distortion_values = kmeans(centroids, dataset)
    
    # Print the final centroids and distortion values
    print("Final centroids:", final_centroids)
    print("Final distortion_values:", distortion_values)

# Entry point to execute the program
if __name__ == "__main__":
    run()
