import pandas as pd
import numpy as np

def load_data():
    return pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

def calculate_means(data):
    population_mean = np.mean(data["Price"])

    april_data = data[data["Month"].astype(str).str.strip().str.lower().isin(["apr", "april"])]

    if len(april_data) == 0:
        return population_mean, None

    april_mean = np.mean(april_data["Price"])

    return population_mean, april_mean

def main():
    data = load_data()

    population_mean, april_mean = calculate_means(data)

    print("Population Mean Price :", population_mean)

    if april_mean is None:
        print("No April data found in the dataset.")
    else:
        print("April Mean Price :", april_mean)

        if april_mean > population_mean:
            print("Observation : April mean is greater than the population mean.")
        elif april_mean < population_mean:
            print("Observation : April mean is less than the population mean.")
        else:
            print("Observation : April mean is equal to the population mean.")

if __name__ == "__main__":
    main()