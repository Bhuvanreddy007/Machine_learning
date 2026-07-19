import pandas as pd
import numpy as np

# Load the thyroid dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    return data

# Identify encoding scheme
def identify_encoding(data):
    encoding_details = []

    for column in data.columns:

        cleaned_column = data[column].replace("?", np.nan)

        numeric_column = pd.to_numeric(cleaned_column, errors="coerce")

        if numeric_column.notna().sum() == cleaned_column.notna().sum():
            continue

        unique_values = set(cleaned_column.dropna().astype(str).str.lower())

        if unique_values.issubset({"t", "f"}):
            encoding = "Label Encoding"
        else:
            encoding = "One-Hot Encoding"

        encoding_details.append((column, encoding))

    return encoding_details

# Main Function
def main():
    data = load_data()

    encoding_details = identify_encoding(data)

    print("Attribute".ljust(35), "Encoding Scheme")
    print("-" * 60)

    for attribute, encoding in encoding_details:
        print(attribute.ljust(35), encoding)

if __name__ == "__main__":
    main()