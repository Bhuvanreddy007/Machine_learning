import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the thyroid dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    return data

# Normalize numeric attributes
def normalize_data(data):

    data = data.replace("?", np.nan)

    numeric_data = pd.DataFrame()

    for column in data.columns:

        numeric_column = pd.to_numeric(data[column], errors="coerce")

        if numeric_column.notna().sum() > 0:

            numeric_data[column] = numeric_column.fillna(numeric_column.median())

    min_max_scaler = MinMaxScaler()
    min_max_data = pd.DataFrame(
        min_max_scaler.fit_transform(numeric_data),
        columns=numeric_data.columns
    )

    standard_scaler = StandardScaler()
    z_score_data = pd.DataFrame(
        standard_scaler.fit_transform(numeric_data),
        columns=numeric_data.columns
    )

    return min_max_data, z_score_data

# Main Function
def main():

    data = load_data()

    min_max_data, z_score_data = normalize_data(data)

    print("Min-Max Normalized Data (First 5 Rows)")
    print("-" * 50)
    print(min_max_data.head())

    print("\nZ-Score Normalized Data (First 5 Rows)")
    print("-" * 50)
    print(z_score_data.head())

if __name__ == "__main__":
    main()