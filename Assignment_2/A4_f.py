import pandas as pd
import numpy as np

# Load the thyroid dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    return data

# Calculate mean and variance for numeric attributes
def calculate_statistics(data):

    statistics_details = []

    for column in data.columns:

        cleaned_column = data[column].replace("?", np.nan)

        numeric_column = pd.to_numeric(cleaned_column, errors="coerce").dropna()

        if len(numeric_column) > 0:

            mean_value = numeric_column.mean()
            variance_value = numeric_column.var()

            statistics_details.append((column, mean_value, variance_value))

    return statistics_details

# Main Function
def main():

    data = load_data()

    statistics_details = calculate_statistics(data)

    print("Attribute".ljust(20), "Mean".ljust(20), "Variance")
    print("-" * 70)

    for attribute, mean_value, variance_value in statistics_details:
        print(attribute.ljust(20),
              f"{mean_value:.2f}".ljust(20),
              f"{variance_value:.2f}")

if __name__ == "__main__":
    main()