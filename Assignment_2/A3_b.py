import pandas as pd
import time

def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
    return data

def calculate_mean(price):
    return sum(price) / len(price)

def calculate_variance(price):
    mean = calculate_mean(price)
    variance = sum((x - mean) ** 2 for x in price) / len(price)
    return variance

def main():
    data = load_data()

    price = data["Price"]

    start = time.time()
    mean = calculate_mean(price)
    variance = calculate_variance(price)
    end = time.time()

    print("Mean:", mean)
    print("Variance:", variance)
    print("Execution Time:", end - start)

if __name__ == "__main__":
    main()