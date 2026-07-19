import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    return pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

def draw_scatter_plot(data):
    plt.figure(figsize=(8,5))
    plt.scatter(data["Day"], data["Chg%"])

    plt.title("Scatter Plot of Chg% vs Day of the Week")
    plt.xlabel("Day of the Week")
    plt.ylabel("Change Percentage (Chg%)")

    plt.grid(True)

    plt.show()

def main():
    data = load_data()
    draw_scatter_plot(data)

if __name__ == "__main__":
    main()