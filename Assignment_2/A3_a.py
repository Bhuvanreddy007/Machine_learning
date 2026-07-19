import pandas as pd
import numpy as np

def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
    return data

def calculate_statistics(price):
    mean = np.mean(price)
    variance = np.var(price)
    return mean, variance

def main():
    data = load_data()

    price = data["Price"]

    mean, variance = calculate_statistics(price)

    print("Mean Price:", mean)
    print("Variance:", variance)

if __name__ == "__main__":
    main()