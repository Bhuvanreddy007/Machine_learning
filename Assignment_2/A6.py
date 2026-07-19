import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the thyroid dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    return data

# Calculate cosine similarity
def calculate_cosine_similarity(data):

    data = data.replace("?", np.nan)

    binary_columns = []

    for column in data.columns:

        unique_values = set(data[column].dropna().astype(str).str.lower())

        if unique_values.issubset({"t", "f"}):
            binary_columns.append(column)

    binary_data = data[binary_columns].replace({"t": 1, "f": 0})

    first_vector = binary_data.iloc[0].values.reshape(1, -1)
    second_vector = binary_data.iloc[1].values.reshape(1, -1)

    similarity = cosine_similarity(first_vector, second_vector)[0][0]

    return similarity

# Main Function
def main():

    data = load_data()

    similarity = calculate_cosine_similarity(data)

    print("Cosine Similarity :", round(similarity, 4))

if __name__ == "__main__":
    main()