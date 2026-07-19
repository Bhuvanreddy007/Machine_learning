import pandas as pd
import numpy as np

# Load the thyroid dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    return data

# Find the range of numeric attributes
def find_numeric_range(data):
    numeric_details = []

    for column in data.columns:

        cleaned_column = data[column].replace("?", np.nan)

        numeric_column = pd.to_numeric(cleaned_column, errors="coerce")

        if numeric_column.notna().sum() == cleaned_column.notna().sum():

            minimum_value = numeric_column.min()
            maximum_value = numeric_column.max()

            numeric_details.append((column, minimum_value, maximum_value))

    return numeric_details

# Main Function
def main():
    data = load_data()

    numeric_details = find_numeric_range(data)

    print("Attribute".ljust(25), "Minimum".ljust(15), "Maximum")
    print("-" * 60)

    for attribute, minimum_value, maximum_value in numeric_details:
        print(attribute.ljust(25), str(minimum_value).ljust(15), maximum_value)

if __name__ == "__main__":
    main()