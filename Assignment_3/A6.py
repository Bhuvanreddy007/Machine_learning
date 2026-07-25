import pandas as pd
from scipy.spatial.distance import minkowski

# Function to load the marketing campaign dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    return data

# Function to calculate Minkowski Distance (Own Function)
def own_minkowski_distance(vector1, vector2, p):

    distance = 0

    # Calculate Minkowski Distance
    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i]) ** p

    distance = distance ** (1 / p)

    return distance

# Function to calculate package Minkowski Distance
def package_minkowski_distance(vector1, vector2, p):

    distance = minkowski(vector1, vector2, p)

    return distance

# Main Function
def main():

    # Load dataset
    data = load_data()

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=["int64", "float64"])

    # Select first two feature vectors
    vector1 = numeric_data.iloc[0].values
    vector2 = numeric_data.iloc[1].values

    p = 2

    # Calculate distances
    own_distance = own_minkowski_distance(vector1, vector2, p)
    package_distance = package_minkowski_distance(vector1, vector2, p)

    # Display results
    print("Own Minkowski Distance      :", round(own_distance, 4))
    print("Package Minkowski Distance  :", round(package_distance, 4))

# Program execution starts here
if __name__ == "__main__":
    main()