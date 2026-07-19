import pandas as pd
import numpy as np

# Load the thyroid dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    return data

# Identify the datatype of each attribute
def identify_data_types(data):
    attribute_details = []

    for column in data.columns:

        # Replace '?' with NaN
        cleaned_column = data[column].replace("?", np.nan)

        # Try converting the column to numeric
        numeric_column = pd.to_numeric(cleaned_column, errors="coerce")

        if numeric_column.notna().sum() > 0 and numeric_column.notna().sum() == cleaned_column.notna().sum():
            data_type = "Numeric"

        else:
            unique_values = set(cleaned_column.dropna().astype(str).str.lower())

            if unique_values.issubset({"t", "f"}):
                data_type = "Binary"
            else:
                data_type = "Nominal"

        attribute_details.append((column, data_type))

    return attribute_details

# Main Function
def main():
    data = load_data()

    attribute_details = identify_data_types(data)

    print("Attribute".ljust(35), "Data Type")
    print("-" * 55)

    for attribute, data_type in attribute_details:
        print(attribute.ljust(35), data_type)

if __name__ == "__main__":
    main()