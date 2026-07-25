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

    return encoded_column, label_mapping

# Function for One-Hot Encoding
def one_hot_encode(column):

    encoded_data = pd.DataFrame()

    unique_values = column.dropna().unique()

    for value in unique_values:
        encoded_data[str(value)] = (column == value).astype(int)

    return encoded_data

# Main Function
def main():

    data = load_data()

    # Label Encoding for Education
    encoded_education, education_mapping = label_encode(data["Education"])

    # One-Hot Encoding for Marital_Status
    encoded_marital = one_hot_encode(data["Marital_Status"])

    print("Label Encoding Mapping (Education)")
    print("-" * 40)
    print(education_mapping)

    print("\nFirst 5 Label Encoded Values")
    print("-" * 40)
    print(encoded_education.head())

    print("\nOne-Hot Encoded Marital_Status (First 5 Rows)")
    print("-" * 50)
    print(encoded_marital.head())

# Program execution starts here
if __name__ == "__main__":
    main()