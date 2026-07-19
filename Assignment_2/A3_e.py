import pandas as pd

def load_data():
    return pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

def calculate_loss_probability(data):
    loss_days = data["Chg%"].apply(lambda x: x < 0).sum()
    total_days = len(data)

    probability = loss_days / total_days

    return loss_days, total_days, probability

def main():
    data = load_data()

    loss_days, total_days, probability = calculate_loss_probability(data)

    print("Total Trading Days :", total_days)
    print("Loss Days :", loss_days)
    print("Probability of Loss :", probability)

if __name__ == "__main__":
    main()