import pandas as pd
import numpy as np

# Load the thyroid dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    return data

# Count missing values in each attribute
def find_missing_values(data):

    data = data.replace("?", np.nan)

    missing_details = []

    for column in data.columns:

        missing_count = data[column].isnull().sum()

        missing_details.append((column, missing_count))

    return missing_details

# Main Function
def main():

    data = load_data()

    missing_details = find_missing_values(data)

    print("Attribute".ljust(35), "Missing Values")
    print("-" * 55)

    for attribute, missing_count in missing_details:
        print(attribute.ljust(35), missing_count)

if __name__ == "__main__":
    main()