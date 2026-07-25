import pandas as pd

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

# Main Function
def main():

    data = load_data()

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=["int64", "float64"])

    # Take first two feature vectors
    vector1 = numeric_data.iloc[0].values
    vector2 = numeric_data.iloc[1].values

    # Manhattan Distance (p = 1)
    manhattan_distance = minkowski_distance(vector1, vector2, 1)

    # Euclidean Distance (p = 2)
    euclidean_distance = minkowski_distance(vector1, vector2, 2)

    print("Manhattan Distance :", round(manhattan_distance, 4))
    print("Euclidean Distance :", round(euclidean_distance, 4))

# Program execution starts here
if __name__ == "__main__":
    main()