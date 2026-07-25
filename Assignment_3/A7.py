import pandas as pd
import numpy as np

# Function to load the marketing campaign dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    return data

# Function to calculate Dot Product
def calculate_dot_product(vector1, vector2):

    dot_product = 0

    # Calculate dot product manually
    for i in range(len(vector1)):
        dot_product += vector1[i] * vector2[i]

    return dot_product

# Function to calculate Euclidean Norm
def calculate_euclidean_norm(vector):

    norm = 0

    # Calculate Euclidean Norm manually
    for value in vector:
        norm += value ** 2

    norm = norm ** 0.5

    return norm

# Main Function
def main():

    # Load dataset
    data = load_data()

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=["int64", "float64"])

    # Select first two feature vectors
    vector1 = numeric_data.iloc[0].values
    vector2 = numeric_data.iloc[1].values

    # Own Functions
    own_dot = calculate_dot_product(vector1, vector2)
    own_norm1 = calculate_euclidean_norm(vector1)
    own_norm2 = calculate_euclidean_norm(vector2)

    # NumPy Functions
    numpy_dot = np.dot(vector1, vector2)
    numpy_norm1 = np.linalg.norm(vector1)
    numpy_norm2 = np.linalg.norm(vector2)

    # Display Results
    print("Own Dot Product        :", round(own_dot, 4))
    print("NumPy Dot Product      :", round(numpy_dot, 4))

    print("\nOwn Euclidean Norm (A) :", round(own_norm1, 4))
    print("NumPy Euclidean Norm(A):", round(numpy_norm1, 4))

    print("\nOwn Euclidean Norm (B) :", round(own_norm2, 4))
    print("NumPy Euclidean Norm(B):", round(numpy_norm2, 4))

# Program execution starts here
if __name__ == "__main__":
    main()