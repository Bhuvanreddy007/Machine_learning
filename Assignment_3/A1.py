import pandas as pd

# Function to load the marketing campaign dataset
def load_data():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    return data

# Function to identify the datatype of each feature
def identify_feature_types(data):

    feature_details = []

    # Check every column in the dataset
    for column in data.columns:

        # Nominal attributes
        if column in ["Education", "Marital_Status"]:
            data_type = "Nominal"

        # Ordinal attributes
        elif column in ["Kidhome", "Teenhome"]:
            data_type = "Ordinal"

        # Interval attribute
        elif column == "Dt_Customer":
            data_type = "Interval"

        # Remaining numeric attributes are Ratio
        else:
            if pd.api.types.is_numeric_dtype(data[column]):
                data_type = "Ratio"
            else:
                data_type = "Nominal"

        # Store feature name and datatype
        feature_details.append((column, data_type))

    return feature_details

# Main Function
def main():

    # Load dataset
    data = load_data()

    # Identify datatypes
    feature_details = identify_feature_types(data)

    # Display the result
    print("Feature".ljust(30), "Data Type")
    print("-" * 50)

    for feature, data_type in feature_details:
        print(feature.ljust(30), data_type)

# Program execution starts here
if __name__ == "__main__":
    main()