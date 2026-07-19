import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
    return data

def prepare_data(data):
    X = data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
    y = data['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')
    return X, y

def train_classifier(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

def main():
    data = load_data()

    X, y = prepare_data(data)

    model = train_classifier(X, y)

    predictions = model.predict(X)

    print("Customer Categories:\n")

    for i in range(len(predictions)):
        print(f"{data['Customer'][i]} : {predictions[i]}")

    print("\nClassifier Model Trained Successfully.")

if __name__ == "__main__":
    main()