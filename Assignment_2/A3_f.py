import pandas as pd

def load_data():
    return pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

def calculate_probability(data):
    data["Day"] = data["Day"].astype(str).str.strip().str.lower()

    wednesday_data = data[data["Day"].isin(["wednesday", "wed"])]

    profit_days = (wednesday_data["Chg%"] > 0).sum()

    total_wednesday = len(wednesday_data)

    probability = profit_days / total_wednesday

    return profit_days, total_wednesday, probability

def main():
    data = load_data()

    profit_days, total_wednesday, probability = calculate_probability(data)

    print("Total Wednesdays :", total_wednesday)
    print("Profit Wednesdays :", profit_days)
    print("Probability of Profit on Wednesday :", probability)

if __name__ == "__main__":
    main()