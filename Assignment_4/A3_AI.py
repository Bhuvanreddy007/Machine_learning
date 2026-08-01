import pandas as pd
import numpy as np
import time

# AI Assisted Code
# Function to load dataset
def load_dataset():

    dataset = pd.read_excel("Lab Session Data.xlsx",
                            sheet_name="marketing_campaign")

    dataset = dataset[["Income", "MntWines"]].dropna()

    return dataset.values


# AI Assisted Code
# Function to calculate Euclidean Distance
def calculate_distance(point, centroid):

    return np.sqrt(np.sum((point - centroid) ** 2))


# AI Assisted Code
# Function to assign clusters
def assign_cluster(data, centroids):

    cluster_labels = []

    for point in data:

        distances = []

        for centroid in centroids:
            distances.append(calculate_distance(point, centroid))

        cluster_labels.append(np.argmin(distances))

    return np.array(cluster_labels)


# AI Assisted Code
# Function to update centroids
def update_centroid(data, labels, k):

    updated_centroids = []

    for cluster in range(k):

        cluster_points = data[labels == cluster]

        if len(cluster_points) == 0:
            updated_centroids.append(data[np.random.randint(len(data))])

        else:
            updated_centroids.append(np.mean(cluster_points, axis=0))

    return np.array(updated_centroids)


# AI Assisted Code
# Function to perform K-Means
def ai_kmeans(data, k, iterations):

    centroids = data[:k].copy()

    for _ in range(iterations):

        labels = assign_cluster(data, centroids)

        new_centroids = update_centroid(data, labels, k)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids


# Main Function
def main():

    dataset = load_dataset()

    k = 3

    iterations = 100

    # Execution Time
    start_time = time.perf_counter()

    ai_kmeans(dataset, k, iterations)

    end_time = time.perf_counter()

    execution_time = end_time - start_time

    print("Execution Time of AI K-Means :", round(execution_time, 6), "seconds")


if __name__ == "__main__":
    main()