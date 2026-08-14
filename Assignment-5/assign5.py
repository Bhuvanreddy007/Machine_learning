# ============================================================
# LAB 5 - HEART DISEASE CLASSIFICATION USING kNN
# File: assign5.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv("heart_disease.csv")

print("\n================ DATASET ================")
print(data.head())

print("\nDataset Shape:", data.shape)
print("\nColumns:")
print(list(data.columns))

print("\nMissing Values:")
print(data.isnull().sum())


# ============================================================
# A1(a) - ENCODING
# ============================================================
# Our dataset is already numerical.
# Therefore, no categorical encoding is required.


# ============================================================
# A1(b) - DATA IMPUTATION
# ============================================================

# Fill missing numerical values using median.
# This will not change anything if there are no missing values.

numeric_columns = data.select_dtypes(include=np.number).columns

for column in numeric_columns:
    data[column] = data[column].fillna(data[column].median())

print("\nMissing values after imputation:")
print(data.isnull().sum())


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]


# ============================================================
# A3 - TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\n================ TRAIN TEST SPLIT ================")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# Convert to NumPy arrays
X_train = X_train.to_numpy(dtype=float)
X_test = X_test.to_numpy(dtype=float)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# ============================================================
# FEATURE SCALING
# ============================================================
# kNN uses distance, so features should be on a similar scale.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# A1(c) - DISTANCE CALCULATION
# ============================================================

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    """
    total = 0

    for i in range(len(point1)):
        difference = point1[i] - point2[i]
        total = total + (difference * difference)

    return np.sqrt(total)


# ============================================================
# A1(d) - SORTING ALGORITHM 1
# BUBBLE SORT
# ============================================================

def bubble_sort(items):
    items = items.copy()

    n = len(items)

    for i in range(n):
        for j in range(0, n - i - 1):

            if items[j][0] > items[j + 1][0]:
                items[j], items[j + 1] = items[j + 1], items[j]

    return items


# ============================================================
# A1(d) - SORTING ALGORITHM 2
# SELECTION SORT
# ============================================================

def selection_sort(items):
    items = items.copy()

    n = len(items)

    for i in range(n):

        minimum = i

        for j in range(i + 1, n):

            if items[j][0] < items[minimum][0]:
                minimum = j

        items[i], items[minimum] = items[minimum], items[i]

    return items


# ============================================================
# A1(d) - SORTING ALGORITHM 3
# INSERTION SORT
# ============================================================

def insertion_sort(items):
    items = items.copy()

    for i in range(1, len(items)):

        current = items[i]
        j = i - 1

        while j >= 0 and items[j][0] > current[0]:

            items[j + 1] = items[j]
            j = j - 1

        items[j + 1] = current

    return items


# ============================================================
# SORTING CONFIGURATION
# ============================================================

def sort_distances(items, method="bubble"):

    if method == "bubble":
        return bubble_sort(items)

    elif method == "selection":
        return selection_sort(items)

    elif method == "insertion":
        return insertion_sort(items)

    else:
        print("Invalid sorting method. Using bubble sort.")
        return bubble_sort(items)


# ============================================================
# A1(e) - IDENTIFY K NEAREST NEIGHBORS
# ============================================================

def find_neighbors(X_train, y_train, test_point, k,
                   sorting_method="bubble"):

    distances = []

    for i in range(len(X_train)):

        distance = calculate_distance(
            X_train[i],
            test_point
        )

        distances.append((distance, y_train[i]))

    distances = sort_distances(
        distances,
        sorting_method
    )

    return distances[:k]


# ============================================================
# A1(f) - MAJORITY VOTING
# ============================================================

def majority_vote(neighbors):

    class_0 = 0
    class_1 = 0

    for distance, label in neighbors:

        if label == 0:
            class_0 += 1
        else:
            class_1 += 1

    # Majority voting
    if class_1 > class_0:
        return 1

    elif class_0 > class_1:
        return 0

    else:
        # Tie-breaking:
        # choose the class of the closest neighbor
        return neighbors[0][1]


# ============================================================
# COMPLETE NORMAL kNN PREDICTION
# ============================================================

def knn_predict_one(
    X_train,
    y_train,
    test_point,
    k,
    sorting_method="bubble"
):

    neighbors = find_neighbors(
        X_train,
        y_train,
        test_point,
        k,
        sorting_method
    )

    prediction = majority_vote(neighbors)

    return prediction


# ============================================================
# A2 - WEIGHTED kNN
# ============================================================

def weighted_vote(neighbors):

    class_0_weight = 0
    class_1_weight = 0

    for distance, label in neighbors:

        # Avoid division by zero
        weight = 1 / (distance + 0.000001)

        if label == 0:
            class_0_weight += weight
        else:
            class_1_weight += weight

    if class_1_weight > class_0_weight:
        return 1

    elif class_0_weight > class_1_weight:
        return 0

    else:
        return neighbors[0][1]


def weighted_knn_predict_one(
    X_train,
    y_train,
    test_point,
    k,
    sorting_method="bubble"
):

    neighbors = find_neighbors(
        X_train,
        y_train,
        test_point,
        k,
        sorting_method
    )

    return weighted_vote(neighbors)


# ============================================================
# PREDICT MULTIPLE TEST SAMPLES
# ============================================================

def custom_knn_predict(
    X_train,
    y_train,
    X_test,
    k,
    sorting_method="bubble"
):

    predictions = []

    for test_point in X_test:

        prediction = knn_predict_one(
            X_train,
            y_train,
            test_point,
            k,
            sorting_method
        )

        predictions.append(prediction)

    return np.array(predictions)


def custom_weighted_knn_predict(
    X_train,
    y_train,
    X_test,
    k,
    sorting_method="bubble"
):

    predictions = []

    for test_point in X_test:

        prediction = weighted_knn_predict_one(
            X_train,
            y_train,
            test_point,
            k,
            sorting_method
        )

        predictions.append(prediction)

    return np.array(predictions)


# ============================================================
# A7 - OUR OWN kNN CLASSIFIER
# ============================================================

class MyKNN:

    def __init__(self, k=3, sorting_method="bubble"):

        self.k = k
        self.sorting_method = sorting_method

        self.X_train = None
        self.y_train = None

    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    def fit(self, X, y):

        self.X_train = np.array(X)
        self.y_train = np.array(y)

        return self

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(self, X):

        predictions = custom_knn_predict(
            self.X_train,
            self.y_train,
            np.array(X),
            self.k,
            self.sorting_method
        )

        return predictions

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    def score(self, X, y):

        predictions = self.predict(X)

        correct = 0

        for i in range(len(y)):

            if predictions[i] == y[i]:
                correct += 1

        accuracy = correct / len(y)

        return accuracy


# ============================================================
# A4 - SKLEARN kNN CLASSIFIER
# k = 3
# ============================================================

print("\n================ A4 - SKLEARN kNN ================")

sklearn_knn = KNeighborsClassifier(
    n_neighbors=3
)

sklearn_knn.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# A5 - SKLEARN ACCURACY
# ============================================================

sklearn_accuracy = sklearn_knn.score(
    X_test_scaled,
    y_test
)

print("Sklearn kNN Accuracy:", sklearn_accuracy)


# ============================================================
# A6 - SKLEARN PREDICTION
# ============================================================

sklearn_predictions = sklearn_knn.predict(
    X_test_scaled
)

print("\nSklearn Predictions:")
print(sklearn_predictions)

print("\nActual Values:")
print(y_test)


# ============================================================
# A1 + A7 - OUR OWN kNN
# ============================================================

print("\n================ OUR OWN kNN ================")

my_knn = MyKNN(
    k=3,
    sorting_method="bubble"
)

my_knn.fit(
    X_train_scaled,
    y_train
)

my_predictions = my_knn.predict(
    X_test_scaled
)

my_accuracy = my_knn.score(
    X_test_scaled,
    y_test
)

print("Sorting method: Bubble Sort")
print("k value:", 3)

print("\nOur Predictions:")
print(my_predictions)

print("\nOur kNN Accuracy:", my_accuracy)


# ============================================================
# TEST ALL THREE SORTING ALGORITHMS
# ============================================================

print("\n================ SORTING COMPARISON ================")

sorting_methods = [
    "bubble",
    "selection",
    "insertion"
]

for method in sorting_methods:

    model = MyKNN(
        k=3,
        sorting_method=method
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    accuracy = model.score(
        X_test_scaled,
        y_test
    )

    print(
        method.capitalize(),
        "Sort Accuracy:",
        accuracy
    )


# ============================================================
# A8 - COMPARISON FOR DIFFERENT k VALUES
# ============================================================

print("\n================ A8 - k COMPARISON ================")

k_values = [1, 3, 5, 7, 9]

my_accuracies = []
sklearn_accuracies = []

# Keep k smaller than the training dataset size

for k in k_values:

    # ---------------- OUR kNN ----------------

    my_model = MyKNN(
        k=k,
        sorting_method="bubble"
    )

    my_model.fit(
        X_train_scaled,
        y_train
    )

    my_accuracy = my_model.score(
        X_test_scaled,
        y_test
    )

    my_accuracies.append(my_accuracy)

    # ---------------- SKLEARN kNN ----------------

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    accuracy = model.score(
        X_test_scaled,
        y_test
    )

    sklearn_accuracies.append(accuracy)

    print(
        "k =", k,
        "| My kNN =", round(my_accuracy, 3),
        "| Sklearn =", round(accuracy, 3)
    )


# ============================================================
# A8 - ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    my_accuracies,
    marker="o",
    label="My kNN"
)

plt.plot(
    k_values,
    sklearn_accuracies,
    marker="s",
    label="Sklearn kNN"
)

plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("kNN Accuracy Comparison")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# A9 - WEIGHTED kNN COMPARISON
# ============================================================

print("\n================ A9 - WEIGHTED kNN ================")

normal_accuracies = []
weighted_accuracies = []

for k in k_values:

    # ---------------- NORMAL kNN ----------------

    normal_predictions = custom_knn_predict(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k,
        "bubble"
    )

    normal_correct = 0

    for i in range(len(y_test)):

        if normal_predictions[i] == y_test[i]:
            normal_correct += 1

    normal_accuracy = normal_correct / len(y_test)

    normal_accuracies.append(normal_accuracy)

    # ---------------- WEIGHTED kNN ----------------

    weighted_predictions = custom_weighted_knn_predict(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k,
        "bubble"
    )

    weighted_correct = 0

    for i in range(len(y_test)):

        if weighted_predictions[i] == y_test[i]:
            weighted_correct += 1

    weighted_accuracy = weighted_correct / len(y_test)

    weighted_accuracies.append(weighted_accuracy)

    print(
        "k =", k,
        "| Normal =", round(normal_accuracy, 3),
        "| Weighted =", round(weighted_accuracy, 3)
    )


# ============================================================
# A9 - WEIGHTED kNN GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    normal_accuracies,
    marker="o",
    label="Normal kNN"
)

plt.plot(
    k_values,
    weighted_accuracies,
    marker="s",
    label="Weighted kNN"
)

plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("Normal kNN vs Weighted kNN")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n================ FINAL RESULTS ================")

best_my_index = np.argmax(my_accuracies)
best_sklearn_index = np.argmax(sklearn_accuracies)
best_weighted_index = np.argmax(weighted_accuracies)

print(
    "Best k for My kNN:",
    k_values[best_my_index]
)

print(
    "Best My kNN Accuracy:",
    my_accuracies[best_my_index]
)

print(
    "Best k for Sklearn:",
    k_values[best_sklearn_index]
)

print(
    "Best Sklearn Accuracy:",
    sklearn_accuracies[best_sklearn_index]
)

print(
    "Best k for Weighted kNN:",
    k_values[best_weighted_index]
)

print(
    "Best Weighted kNN Accuracy:",
    weighted_accuracies[best_weighted_index]
)

print("\n================ PROGRAM COMPLETED ================")