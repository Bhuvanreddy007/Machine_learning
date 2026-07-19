import pandas as pd
import numpy as np

# Load the thyroid dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    return data

# Find outliers using IQR method
def detect_outliers(data):

    outlier_details = []

    for column in data.columns:

        cleaned_column = data[column].replace("?", np.nan)

        numeric_column = pd.to_numeric(cleaned_column, errors="coerce").dropna()

        if len(numeric_column) > 0:

            Q1 = numeric_column.quantile(0.25)
            Q3 = numeric_column.quantile(0.75)

            IQR = Q3 - Q1

            lower_limit = Q1 - (1.5 * IQR)
            upper_limit = Q3 + (1.5 * IQR)

            outliers = numeric_column[
                (numeric_column < lower_limit) |
                (numeric_column > upper_limit)
            ]

            outlier_details.append((column, len(outliers)))

    return outlier_details

# Main Function
def main():

    data = load_data()

    outlier_details = detect_outliers(data)

    print("Attribute".ljust(25), "Number of Outliers")
    print("-" * 50)

    for attribute, count in outlier_details:
        print(attribute.ljust(25), count)

if __name__ == "__main__":
    main()