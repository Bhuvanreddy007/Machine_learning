import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to load the marketing campaign dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    return data

# Function to get a numeric feature
def get_feature_data(data):

    feature = pd.to_numeric(data["Income"], errors="coerce").dropna()

    return feature

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

# Function to plot Histogram
def plot_histogram(feature):

    plt.figure(figsize=(8,5))

    plt.hist(feature, bins=10)

    plt.title("Histogram of Income")

    plt.xlabel("Income")

    plt.ylabel("Frequency")

    plt.grid(True)

    plt.show()

# Main Function
def main():

    data = load_data()

    feature = get_feature_data(data)

    mean = calculate_mean(feature.tolist())

    variance = calculate_variance(feature.tolist())

    print("Feature : Income")
    print("Mean :", round(mean,2))
    print("Variance :", round(variance,2))

    plot_histogram(feature)

# Program execution starts here
if __name__ == "__main__":
    main()