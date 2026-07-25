import pandas as pd

# Function to load the marketing campaign dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    return data

# Function to calculate Mean
def calculate_mean(values):

    total = 0

    for value in values:
        total += value

    mean = total / len(values)

    return mean

# Function to calculate Variance
def calculate_variance(values):

    mean = calculate_mean(values)

    total = 0

    for value in values:
        total += (value - mean) ** 2

    variance = total / len(values)

    return variance

# Function to calculate Standard Deviation
def calculate_standard_deviation(values):

    variance = calculate_variance(values)

    standard_deviation = variance ** 0.5

    return standard_deviation

# Function to calculate statistics for all numeric features
def calculate_dataset_statistics(data):

    numeric_data = data.select_dtypes(include=["int64", "float64"])

    statistics = []

    for column in numeric_data.columns:

        values = numeric_data[column].dropna().tolist()

        mean = calculate_mean(values)
        variance = calculate_variance(values)
        standard_deviation = calculate_standard_deviation(values)

        statistics.append(
            (column, mean, variance, standard_deviation)
        )

    return statistics

# Main Function
def main():

    data = load_data()

    statistics = calculate_dataset_statistics(data)

    print("Feature".ljust(25),
          "Mean".ljust(15),
          "Variance".ljust(20),
          "Standard Deviation")

    print("-" * 85)

    for feature, mean, variance, standard_deviation in statistics:

        print(feature.ljust(25),
              f"{mean:.2f}".ljust(15),
              f"{variance:.2f}".ljust(20),
              f"{standard_deviation:.2f}")

# Program execution starts here
if __name__ == "__main__":
    main()