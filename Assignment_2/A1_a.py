import pandas as pd
import numpy as np

def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
    X = data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    return X

def calculate_rank(X):
    return np.linalg.matrix_rank(X)

def main():
    X = load_data()
    rank = calculate_rank(X)

    print("Feature Matrix (X):")
    print(X)
    print("\nRank of Feature Matrix:", rank)

if __name__ == "__main__":
    main()