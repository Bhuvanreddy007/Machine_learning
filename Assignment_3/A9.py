import pandas as pd
import numpy as np

# Function to load the marketing campaign dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    return data

# Function to calculate Mean using NumPy
def calculate_numpy_mean(data):

    numeric_data = data.select_dtypes(include=["int64", "float64"])

    mean_values = np.mean(numeric_data, axis=0)

    return mean_values

# Function to calculate Standard Deviation using NumPy
def calculate_numpy_standard_deviation(data):

    numeric_data = data.select_dtypes(include=["int64", "float64"])

    standard_deviation_values = np.std(numeric_data, axis=0)

    return standard_deviation_values

# Main Function
def main():

    # Load dataset
    data = load_data()

    # Calculate statistics
    mean_values = calculate_numpy_mean(data)
    standard_deviation_values = calculate_numpy_standard_deviation(data)

    # Display results
    print("Feature".ljust(25),
          "NumPy Mean".ljust(15),
          "NumPy Std Dev")

    print("-" * 60)

    for feature, mean, std in zip(mean_values.index,
                                  mean_values.values,
                                  standard_deviation_values.values):

        print(feature.ljust(25),
              f"{mean:.2f}".ljust(15),
              f"{std:.2f}")

# Program execution starts here
if __name__ == "__main__":
    main()