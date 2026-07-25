import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to load the marketing campaign dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    return data

# Function to calculate Euclidean Distance
def euclidean_distance(point1, point2):

    distance = 0

    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2

    return distance ** 0.5

# Function to assign data points to nearest centroid
def assign_clusters(data, centroids):

    clusters = []

    for point in data:

        distances = []

        for centroid in centroids:
            distances.append(euclidean_distance(point, centroid))

        clusters.append(np.argmin(distances))

    return np.array(clusters)

# Function to update centroids
def update_centroids(data, clusters, k):

    centroids = []

    for cluster in range(k):

        cluster_points = data[clusters == cluster]

        if len(cluster_points) > 0:
            centroids.append(cluster_points.mean(axis=0))
        else:
            centroids.append(data[np.random.randint(len(data))])

    return np.array(centroids)

# Function to perform K-Means
def kmeans(data, k, iterations):

    centroids = data[:k].copy()

    for i in range(iterations):

        clusters = assign_clusters(data, centroids)

        new_centroids = update_centroids(data, clusters, k)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids

# Function to plot clusters
def plot_clusters(data, clusters, centroids):

    plt.figure(figsize=(8,6))

    plt.scatter(data[:,0], data[:,1], c=clusters)

    plt.scatter(
        centroids[:,0],
        centroids[:,1],
        marker="X",
        s=200
    )

    plt.title("K-Means Clustering")

    plt.xlabel("Income")

    plt.ylabel("MntWines")

    plt.grid(True)

    plt.show()

# Main Function
def main():

    data = load_data()

    numeric_data = data[["Income","MntWines"]].dropna()

    feature_matrix = numeric_data.values

    k = 3

    iterations = 100

    clusters, centroids = kmeans(feature_matrix, k, iterations)

    print("Final Centroids")
    print("----------------")

    print(centroids)

    plot_clusters(feature_matrix, clusters, centroids)

# Program execution starts here
if __name__ == "__main__":
    main()