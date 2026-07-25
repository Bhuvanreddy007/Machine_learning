import pandas as pd

# Function to load the marketing campaign dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    return data

# Function for Label Encoding
def label_encode(column):

    unique_values = column.dropna().unique()

    label_mapping = {}

    for index, value in enumerate(unique_values):
        label_mapping[value] = index

    encoded_column = column.map(label_mapping)

    return encoded_column

# Function for One-Hot Encoding
def one_hot_encode(column):

    encoded_data = pd.DataFrame()

    unique_values = column.dropna().unique()

    for value in unique_values:
        encoded_data[column.name + "_" + str(value)] = (column == value).astype(int)

    return encoded_data

# Function to encode categorical features
def encode_dataset(data):

    encoded_dataset = data.copy()

    # Label Encoding for Education
    encoded_dataset["Education"] = label_encode(encoded_dataset["Education"])

    # One-Hot Encoding for Marital_Status
    marital_encoded = one_hot_encode(encoded_dataset["Marital_Status"])

    # Remove original categorical column
    encoded_dataset = encoded_dataset.drop(columns=["Marital_Status"])

    # Add One-Hot Encoded columns
    encoded_dataset = pd.concat([encoded_dataset, marital_encoded], axis=1)

    return encoded_dataset

# Main Function
def main():

    data = load_data()

    encoded_dataset = encode_dataset(data)

    print("Original Dataset Shape")
    print("----------------------")
    print(data.shape)

    print("\nEncoded Dataset Shape")
    print("----------------------")
    print(encoded_dataset.shape)

    print("\nFirst 5 Rows of Encoded Dataset")
    print("--------------------------------")
    print(encoded_dataset.head())

# Program execution starts here
if __name__ == "__main__":
    main()