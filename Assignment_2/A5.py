import pandas as pd
import numpy as np

# Load the thyroid dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    return data

# Calculate Jaccard Coefficient and Simple Matching Coefficient
def calculate_similarity(data):

    data = data.replace("?", np.nan)

    binary_columns = []

    for column in data.columns:

        unique_values = set(data[column].dropna().astype(str).str.lower())

        if unique_values.issubset({"t", "f"}):
            binary_columns.append(column)

    first_vector = data.loc[0, binary_columns].replace({"t": 1, "f": 0}).astype(int)
    second_vector = data.loc[1, binary_columns].replace({"t": 1, "f": 0}).astype(int)

    f11 = np.sum((first_vector == 1) & (second_vector == 1))
    f10 = np.sum((first_vector == 1) & (second_vector == 0))
    f01 = np.sum((first_vector == 0) & (second_vector == 1))
    f00 = np.sum((first_vector == 0) & (second_vector == 0))

    jaccard = f11 / (f11 + f10 + f01)

    smc = (f11 + f00) / (f11 + f10 + f01 + f00)

    return jaccard, smc

# Main Function
def main():

    data = load_data()

    jaccard, smc = calculate_similarity(data)

    print("Jaccard Coefficient :", round(jaccard, 4))
    print("Simple Matching Coefficient :", round(smc, 4))

if __name__ == "__main__":
    main()