import pandas as pd
import matplotlib.pyplot as plt

# Function to load the marketing campaign dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    return data

# Function to calculate Minkowski Distance
def minkowski_distance(vector1, vector2, p):

    distance = 0

    # Calculate Minkowski distance
    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i]) ** p

    distance = distance ** (1 / p)

    return distance

# Function to calculate distance for p = 1 to 10
def calculate_distances(vector1, vector2):

    p_values = []
    distances = []

    for p in range(1, 11):

        p_values.append(p)

        distance = minkowski_distance(vector1, vector2, p)

        distances.append(distance)

    return p_values, distances

# Function to plot graph
def plot_graph(p_values, distances):

    plt.figure(figsize=(8,5))

    plt.plot(p_values, distances, marker="o")

    plt.title("Minkowski Distance for Different p Values")
    plt.xlabel("Order (p)")
    plt.ylabel("Distance")

    plt.grid(True)

    plt.show()

# Main Function
def main():

    data = load_data()

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=["int64", "float64"])

    # Take first two feature vectors
    vector1 = numeric_data.iloc[0].values
    vector2 = numeric_data.iloc[1].values

    p_values, distances = calculate_distances(vector1, vector2)

    print("p".ljust(10), "Distance")
    print("-" * 25)

    for p, distance in zip(p_values, distances):
        print(str(p).ljust(10), round(distance, 4))

    plot_graph(p_values, distances)

# Program execution starts here
if __name__ == "__main__":
    main()