import pandas as pd
import numpy as np

def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")

    X = data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    y = data[['Payment (Rs)']].values

    return X, y

def calculate_product_cost(X, y):
    X_pseudo = np.linalg.pinv(X)
    cost = X_pseudo @ y
    return cost

def main():
    X, y = load_data()

    cost = calculate_product_cost(X, y)

    print("Estimated Cost of Products")
    print("Candy Cost      :", cost[0][0])
    print("Mango Cost      :", cost[1][0])
    print("Milk Packet Cost:", cost[2][0])

if __name__ == "__main__":
    main()