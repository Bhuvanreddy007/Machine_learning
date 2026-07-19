import pandas as pd

def load_data():
    return pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

def calculate_conditional_probability(data):
    data["Day"] = data["Day"].astype(str).str.strip().str.lower()

    wednesday_data = data[data["Day"].isin(["wednesday", "wed"])]

    total_wednesday = len(wednesday_data)

    profit_wednesday = (wednesday_data["Chg%"] > 0).sum()

    conditional_probability = profit_wednesday / total_wednesday

    return total_wednesday, profit_wednesday, conditional_probability

def main():
    data = load_data()

    total_wednesday, profit_wednesday, conditional_probability = calculate_conditional_probability(data)

    print("Total Wednesdays :", total_wednesday)
    print("Profit Wednesdays :", profit_wednesday)
    print("Conditional Probability of Profit given Wednesday :", conditional_probability)

if __name__ == "__main__":
    main()