import pandas as pd
import numpy as np

# Load the thyroid dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    return data

# Impute missing values
def impute_missing_values(data):

    data = data.replace("?", np.nan)
    imputed_data = data.copy()

    # Numeric columns having outliers (from A4(e))
    outlier_columns = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]

    for column in imputed_data.columns:

        cleaned_column = imputed_data[column]

        numeric_column = pd.to_numeric(cleaned_column, errors="coerce")

        if numeric_column.notna().sum() > 0:

            # Use Median for numeric columns with outliers
            if column in outlier_columns:
                median_value = numeric_column.median()
                imputed_data[column] = numeric_column.fillna(median_value)

            # Use Mean for numeric columns without outliers
            else:
                mean_value = numeric_column.mean()
                imputed_data[column] = numeric_column.fillna(mean_value)

        else:
            # Use Mode for categorical/binary columns
            mode_value = cleaned_column.mode()[0]
            imputed_data[column] = cleaned_column.fillna(mode_value)

    return imputed_data

# Count remaining missing values
def count_missing_values(data):

    missing_details = []

    for column in data.columns:
        missing_count = data[column].isnull().sum()
        missing_details.append((column, missing_count))

    return missing_details

# Main Function
def main():

    data = load_data()

    imputed_data = impute_missing_values(data)

    missing_details = count_missing_values(imputed_data)

    print("Attribute".ljust(35), "Remaining Missing Values")
    print("-" * 60)

    for attribute, missing_count in missing_details:
        print(attribute.ljust(35), missing_count)

if __name__ == "__main__":
    main()