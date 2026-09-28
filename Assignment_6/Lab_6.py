# ============================================================
# LAB 6 - HEART DISEASE CLASSIFICATION USING kNN
# File: Lab_6_Heart_Disease_kNN.py
# Subject: 23CSE301
# ============================================================
#
# Dataset:
# Heart Disease Dataset
#
# Lab-6 Main Requirements:
# A1. Repeat the Lab-5 experiments using the Lab-5 dataset and an AI tool.
# A2. Generate unit test cases for modular functions.
# A3. Compare three kNN versions:
#     1. Our own kNN
#     2. Scikit-Learn kNN
#     3. GenAI-developed kNN
#     Evaluate Accuracy, Precision, Recall, F-score and
#     average computational time over 10 runs.
#
# GenAI Tool Used:
# ChatGPT
# ============================================================

import time
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# LOAD DATASET
# ============================================================
# GenAI Tool Used: ChatGPT
# The dataset is the Heart Disease Dataset used in Lab-5
# and reused for Lab-6 as required.

DATA_FILE = "heart_disease.csv"

data = pd.read_csv(DATA_FILE)

print("\n================ DATASET ================")
print(data)

print("\nDataset Shape:", data.shape)

print("\nColumns:")
print(list(data.columns))

print("\nMissing Values:")
print(data.isnull().sum())


# ============================================================
# A1(a) - ENCODING
# ============================================================
# GenAI Tool Used: ChatGPT
#
# The supplied Heart Disease dataset is already numerical.
# Therefore, no categorical feature encoding is required.
# The target column HeartDisease is already encoded as 0/1.


# ============================================================
# A1(b) - DATA IMPUTATION
# ============================================================
# GenAI Tool Used: ChatGPT
#
# Fill missing numerical values using median.
# This does not change complete columns.

numeric_columns = data.select_dtypes(include=np.number).columns

for column in numeric_columns:
    data[column] = data[column].fillna(data[column].median())

print("\nMissing values after imputation:")
print(data.isnull().sum())


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

feature_columns = [
    "Age",
    "Sex",
    "ChestPain",
    "BloodPressure",
    "Cholesterol",
    "MaxHeartRate",
    "ExerciseAngina"
]

X = data[feature_columns]

y = data["HeartDisease"]


# ============================================================
# A3 - TRAIN TEST SPLIT
# ============================================================
# GenAI Tool Used: ChatGPT
#
# The same 70:30 split, random_state=42 and stratification
# used in Lab-5 are retained for the Lab-6 experiments.

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
y_train = y_train.to_numpy(dtype=int)
y_test = y_test.to_numpy(dtype=int)


# ============================================================
# FEATURE SCALING
# ============================================================
# GenAI Tool Used: ChatGPT
#
# kNN is distance-based. StandardScaler prevents the Payment
# feature from dominating the distance calculation.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# A1(c) - DISTANCE CALCULATION
# ============================================================
# GenAI Tool Used: ChatGPT
#
# Euclidean distance is calculated manually to demonstrate the
# basic operation used by kNN.

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    """
    total = 0.0

    for i in range(len(point1)):
        difference = point1[i] - point2[i]
        total = total + (difference * difference)

    return np.sqrt(total)


# ============================================================
# A1(d) - SORTING ALGORITHM 1
# BUBBLE SORT
# ============================================================
# GenAI Tool Used: ChatGPT

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
# GenAI Tool Used: ChatGPT

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
# GenAI Tool Used: ChatGPT

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
# GenAI Tool Used: ChatGPT

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
# GenAI Tool Used: ChatGPT

def find_neighbors(
    X_train,
    y_train,
    test_point,
    k,
    sorting_method="bubble"
):

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
# GenAI Tool Used: ChatGPT

def majority_vote(neighbors):

    class_0 = 0
    class_1 = 0

    for distance, label in neighbors:

        if label == 0:
            class_0 += 1
        else:
            class_1 += 1

    if class_1 > class_0:
        return 1

    elif class_0 > class_1:
        return 0

    else:
        # Tie-breaking: use the class of the closest neighbor.
        return neighbors[0][1]


# ============================================================
# COMPLETE NORMAL kNN PREDICTION
# ============================================================
# GenAI Tool Used: ChatGPT

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

    return majority_vote(neighbors)


# ============================================================
# A2 - WEIGHTED kNN
# ============================================================
# GenAI Tool Used: ChatGPT
#
# Closer neighbours receive larger weights.

def weighted_vote(neighbors):

    class_0_weight = 0.0
    class_1_weight = 0.0

    for distance, label in neighbors:

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
# GenAI Tool Used: ChatGPT

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
# GenAI Tool Used: ChatGPT
#
# This class provides fit(), predict(), and score() methods
# similar to a standard machine-learning classifier.

class MyKNN:

    def __init__(self, k=3, sorting_method="bubble"):

        self.k = k
        self.sorting_method = sorting_method

        self.X_train = None
        self.y_train = None

    def fit(self, X, y):

        self.X_train = np.array(X)
        self.y_train = np.array(y)

        return self

    def predict(self, X):

        predictions = custom_knn_predict(
            self.X_train,
            self.y_train,
            np.array(X),
            self.k,
            self.sorting_method
        )

        return predictions

    def score(self, X, y):

        predictions = self.predict(X)

        correct = 0

        for i in range(len(y)):

            if predictions[i] == y[i]:
                correct += 1

        return correct / len(y)


# ============================================================
# A4 - SKLEARN kNN CLASSIFIER
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
# GenAI Tool Used: ChatGPT
#
# The Lab-5 code used k = 1, 3, 5, 7, 9.
# The Heart Disease dataset has 14 training samples after
# the 70:30 split, so all Lab-5 k values remain valid.

print("\n================ A8 - k COMPARISON ================")

k_values = [1, 3, 5, 7, 9]

my_accuracies = []
sklearn_accuracies = []

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

    normal_predictions = custom_knn_predict(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k,
        "bubble"
    )

    normal_accuracy = accuracy_score(
        y_test,
        normal_predictions
    )

    normal_accuracies.append(normal_accuracy)

    weighted_predictions = custom_weighted_knn_predict(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k,
        "bubble"
    )

    weighted_accuracy = accuracy_score(
        y_test,
        weighted_predictions
    )

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
# A2 - UNIT TESTS FOR LAB-5 FUNCTIONS
# ============================================================
# GenAI Tool Used: ChatGPT
#
# These tests cover the modular functions used in the
# Lab-5-style experiments adapted to this dataset.

class TestLab5Functions(unittest.TestCase):

    def setUp(self):

        self.items = [
            (3.0, 1),
            (1.0, 0),
            (2.0, 1)
        ]

        self.X_train_test = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [10.0, 11.0]
        ])

        self.y_train_test = np.array([0, 0, 1, 1])

    def test_calculate_distance(self):

        result = calculate_distance(
            np.array([0.0, 0.0]),
            np.array([3.0, 4.0])
        )

        self.assertAlmostEqual(result, 5.0)

    def test_bubble_sort(self):

        result = bubble_sort(self.items)

        self.assertEqual(
            result,
            [(1.0, 0), (2.0, 1), (3.0, 1)]
        )

    def test_selection_sort(self):

        result = selection_sort(self.items)

        self.assertEqual(
            result,
            [(1.0, 0), (2.0, 1), (3.0, 1)]
        )

    def test_insertion_sort(self):

        result = insertion_sort(self.items)

        self.assertEqual(
            result,
            [(1.0, 0), (2.0, 1), (3.0, 1)]
        )

    def test_sort_distances(self):

        for method in [
            "bubble",
            "selection",
            "insertion"
        ]:

            result = sort_distances(
                self.items,
                method
            )

            self.assertEqual(
                result,
                [(1.0, 0), (2.0, 1), (3.0, 1)]
            )

    def test_find_neighbors(self):

        neighbors = find_neighbors(
            self.X_train_test,
            self.y_train_test,
            np.array([0.0, 0.2]),
            k=3
        )

        self.assertEqual(len(neighbors), 3)
        self.assertEqual(neighbors[0][1], 0)

    def test_majority_vote(self):

        neighbors = [
            (0.1, 0),
            (0.2, 0),
            (0.3, 1)
        ]

        self.assertEqual(
            majority_vote(neighbors),
            0
        )

    def test_majority_vote_tie(self):

        neighbors = [
            (0.1, 1),
            (0.2, 0)
        ]

        self.assertEqual(
            majority_vote(neighbors),
            1
        )

    def test_weighted_vote(self):

        neighbors = [
            (0.1, 1),
            (1.0, 0),
            (1.5, 0)
        ]

        self.assertEqual(
            weighted_vote(neighbors),
            1
        )

    def test_knn_predict_one(self):

        prediction = knn_predict_one(
            self.X_train_test,
            self.y_train_test,
            np.array([0.0, 0.2]),
            k=3
        )

        self.assertEqual(prediction, 0)

    def test_weighted_knn_predict_one(self):

        prediction = weighted_knn_predict_one(
            self.X_train_test,
            self.y_train_test,
            np.array([10.0, 10.2]),
            k=3
        )

        self.assertEqual(prediction, 1)

    def test_custom_knn_predict(self):

        X_test_unit = np.array([
            [0.0, 0.2],
            [10.0, 10.2]
        ])

        predictions = custom_knn_predict(
            self.X_train_test,
            self.y_train_test,
            X_test_unit,
            k=3
        )

        self.assertEqual(
            predictions.tolist(),
            [0, 1]
        )

    def test_custom_weighted_knn_predict(self):

        X_test_unit = np.array([
            [0.0, 0.2],
            [10.0, 10.2]
        ])

        predictions = custom_weighted_knn_predict(
            self.X_train_test,
            self.y_train_test,
            X_test_unit,
            k=3
        )

        self.assertEqual(
            predictions.tolist(),
            [0, 1]
        )

    def test_my_knn_class(self):

        model = MyKNN(
            k=3,
            sorting_method="bubble"
        )

        model.fit(
            self.X_train_test,
            self.y_train_test
        )

        predictions = model.predict(
            np.array([[0.0, 0.2], [10.0, 10.2]])
        )

        self.assertEqual(
            predictions.tolist(),
            [0, 1]
        )

        self.assertEqual(
            model.score(
                np.array([[0.0, 0.2], [10.0, 10.2]]),
                np.array([0, 1])
            ),
            1.0
        )


# ============================================================
# A2 - UNIT TESTS FOR LAB-6 FUNCTIONS
# ============================================================
# GenAI Tool Used: ChatGPT
#
# These tests cover the additional functions used for
# standardization and the Lab-6 performance evaluation.

def calculate_classification_metrics(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1-Score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }


def standardize_train_test(X_train_local, X_test_local):

    local_scaler = StandardScaler()

    X_train_local = local_scaler.fit_transform(
        X_train_local
    )

    X_test_local = local_scaler.transform(
        X_test_local
    )

    return X_train_local, X_test_local


# ============================================================
# A3 - GENAI-DEVELOPED kNN
# ============================================================
# GenAI Tool Used: ChatGPT
#
# This is a separate vectorized implementation. It uses
# NumPy norm calculations instead of the Lab-5-style manual
# distance loop and sorting functions.

def genai_knn_predict(
    X_train,
    y_train,
    X_test,
    k=3
):

    predictions = []

    for test_point in X_test:

        distances = np.linalg.norm(
            X_train - test_point,
            axis=1
        )

        nearest_indices = np.argsort(
            distances
        )[:k]

        nearest_labels = y_train[
            nearest_indices
        ]

        values, counts = np.unique(
            nearest_labels,
            return_counts=True
        )

        prediction = values[
            np.argmax(counts)
        ]

        predictions.append(prediction)

    return np.array(predictions)


class TestLab6Functions(unittest.TestCase):

    def test_standardize_train_test(self):

        X_train_unit = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0]
        ])

        X_test_unit = np.array([
            [2.0, 20.0]
        ])

        X_train_scaled, X_test_scaled = (
            standardize_train_test(
                X_train_unit,
                X_test_unit
            )
        )

        self.assertTrue(
            np.allclose(
                X_train_scaled.mean(axis=0),
                0.0
            )
        )

        self.assertEqual(
            X_test_scaled.shape,
            (1, 2)
        )

    def test_classification_metrics(self):

        y_true_unit = np.array([0, 0, 1, 1])
        y_pred_unit = np.array([0, 0, 1, 1])

        metrics = calculate_classification_metrics(
            y_true_unit,
            y_pred_unit
        )

        self.assertEqual(
            metrics["Accuracy"],
            1.0
        )

        self.assertEqual(
            metrics["Precision"],
            1.0
        )

        self.assertEqual(
            metrics["Recall"],
            1.0
        )

        self.assertEqual(
            metrics["F1-Score"],
            1.0
        )

    def test_genai_knn(self):

        X_train_unit = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [10.0, 11.0]
        ])

        y_train_unit = np.array([0, 0, 1, 1])

        X_test_unit = np.array([
            [0.0, 0.2],
            [10.0, 10.2]
        ])

        predictions = genai_knn_predict(
            X_train_unit,
            y_train_unit,
            X_test_unit,
            k=3
        )

        self.assertEqual(
            predictions.tolist(),
            [0, 1]
        )


# ============================================================
# RUN UNIT TESTS
# ============================================================

def run_unit_tests():

    print("\n================ A2 - UNIT TESTS ================")

    suite = unittest.TestSuite()

    suite.addTests(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            TestLab5Functions
        )
    )

    suite.addTests(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            TestLab6Functions
        )
    )

    runner = unittest.TextTestRunner(
        verbosity=2
    )

    result = runner.run(suite)

    print("\nUnit tests executed:", result.testsRun)
    print("Failures:", len(result.failures))
    print("Errors:", len(result.errors))

    return result.wasSuccessful()


# ============================================================
# A3 - PERFORMANCE COMPARISON
# ============================================================
# GenAI Tool Used: ChatGPT
#
# The three required versions are:
# 1. Our own Lab-5-style kNN
# 2. Scikit-Learn kNN
# 3. GenAI-developed vectorized kNN
#
# Each version is executed 10 times and average computational
# time is reported.

def evaluate_custom_knn():

    return custom_knn_predict(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k=3,
        sorting_method="bubble"
    )


def evaluate_sklearn_knn():

    model = KNeighborsClassifier(
        n_neighbors=3
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    return model.predict(
        X_test_scaled
    )


def evaluate_genai_knn():

    return genai_knn_predict(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k=3
    )


def performance_test(
    algorithm_name,
    prediction_function,
    runs=10
):

    execution_times = []

    predictions = None

    for _ in range(runs):

        start_time = time.perf_counter()

        predictions = prediction_function()

        end_time = time.perf_counter()

        execution_times.append(
            end_time - start_time
        )

    metrics = calculate_classification_metrics(
        y_test,
        predictions
    )

    return {
        "Algorithm": algorithm_name,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1-Score": metrics["F1-Score"],
        "Average Time (sec)": np.mean(
            execution_times
        )
    }


def run_performance_comparison():

    print(
        "\n================ A3 - PERFORMANCE COMPARISON ================"
    )

    results = []

    results.append(
        performance_test(
            "Our Own kNN",
            evaluate_custom_knn,
            runs=10
        )
    )

    results.append(
        performance_test(
            "Scikit-Learn kNN",
            evaluate_sklearn_knn,
            runs=10
        )
    )

    results.append(
        performance_test(
            "GenAI kNN",
            evaluate_genai_knn,
            runs=10
        )
    )

    results_df = pd.DataFrame(results)

    print(
        "\nPerformance Comparison Table:\n"
    )

    print(
        results_df.to_string(index=False)
    )

    results_df.to_csv(
        "Lab6_KNN_Performance_Results.csv",
        index=False
    )

    return results_df


# ============================================================
# FINAL RESULTS
# ============================================================

def print_final_summary():

    print("\n================ FINAL SUMMARY ================")

    print(
        "Dataset:",
        DATA_FILE
    )

    print(
        "Number of records:",
        len(data)
    )

    print(
        "Number of features:",
        len(feature_columns)
    )

    print(
        "Target:",
        "HeartDisease"
    )

    print(
        "k used for A3:",
        3
    )

    print(
        "Train-test split:",
        "70:30"
    )

    print(
        "Performance runs:",
        10
    )

    print("\nLab-6 execution completed.")


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n============================================================")
    print("        LAB 6 - HEART DISEASE DATASET")
    print("============================================================")

    tests_passed = run_unit_tests()

    results_df = run_performance_comparison()

    print_final_summary()

    if tests_passed:
        print("\nAll unit tests passed successfully.")
    else:
        print("\nSome unit tests failed. Check the test output above.")

    print(
        "\nPerformance results were saved to:"
        " Lab6_KNN_Performance_Results.csv"
    )
