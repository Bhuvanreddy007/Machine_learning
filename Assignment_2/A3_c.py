import pandas as pd
import numpy as np

def load_data():
    return pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

def calculate_means(data):
    data["Day"] = data["Day"].astype(str).str.strip().str.lower()

    population_mean = np.mean(data["Price"])

    wednesday_data = data[data["Day"].isin(["wednesday", "wed"])]

    if len(wednesday_data) == 0:
        return population_mean, None

    wednesday_mean = np.mean(wednesday_data["Price"])

    return population_mean, wednesday_mean

def main():
    data = load_data()

    population_mean, wednesday_mean = calculate_means(data)

    print("Population Mean Price :", population_mean)

    if wednesday_mean is None:
        print("No Wednesday data found in the dataset.")
    else:
        print("Wednesday Mean Price :", wednesday_mean)

        if wednesday_mean > population_mean:
            print("Observation : Wednesday mean is greater than the population mean.")
        elif wednesday_mean < population_mean:
            print("Observation : Wednesday mean is less than the population mean.")
        else:
            print("Observation : Wednesday mean is equal to the population mean.")

if __name__ == "__main__":
    main()