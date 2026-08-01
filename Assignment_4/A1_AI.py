import pandas as pd

# AI Assisted Code
# Function to import dataset
def import_dataset():

    excel_file = "Lab Session Data.xlsx"
    worksheet = "marketing_campaign"

    dataset = pd.read_excel(excel_file, sheet_name=worksheet)

    return dataset


# AI Assisted Code
# Function to classify attributes
def categorize_features(dataset):

    feature_information = {}

    for feature in dataset.columns:

        if feature == "Dt_Customer":
            feature_information[feature] = "Interval"

        elif feature in ["Education", "Marital_Status"]:
            feature_information[feature] = "Nominal"

        elif feature in ["Kidhome", "Teenhome"]:
            feature_information[feature] = "Ordinal"

        else:
            feature_information[feature] = "Ratio"

    return feature_information


# AI Assisted Code
# Function to display results
def display_feature_information(feature_information):

    print("Feature".ljust(30), "Data Type")
    print("-" * 45)

    for feature, datatype in feature_information.items():
        print(feature.ljust(30), datatype)


# Main Function
def main():

    marketing_data = import_dataset()

    feature_information = categorize_features(marketing_data)

    display_feature_information(feature_information)


if __name__ == "__main__":
    main()