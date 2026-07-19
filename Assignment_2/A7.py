import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the thyroid dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    return data

# Prepare numeric data and calculate correlation
def calculate_correlation(data):

    data = data.replace("?", np.nan)

    numeric_data = pd.DataFrame()

    for column in data.columns:

        numeric_column = pd.to_numeric(data[column], errors="coerce")

        if numeric_column.notna().sum() > 0:
            numeric_data[column] = numeric_column

    correlation_matrix = numeric_data.corr()

    return correlation_matrix

# Draw heatmap
def draw_heatmap(correlation_matrix):

    plt.figure(figsize=(8,6))

    plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")

    plt.colorbar(label="Correlation")

    plt.xticks(range(len(correlation_matrix.columns)),
               correlation_matrix.columns,
               rotation=90)

    plt.yticks(range(len(correlation_matrix.columns)),
               correlation_matrix.columns)

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.show()

# Main Function
def main():

    data = load_data()

    correlation_matrix = calculate_correlation(data)

    print("Correlation Matrix:\n")
    print(correlation_matrix)

    draw_heatmap(correlation_matrix)

if __name__ == "__main__":
    main()