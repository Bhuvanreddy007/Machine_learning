import unittest
import numpy as np

# AI Assisted Code
# Function to calculate Minkowski Distance
def minkowski_distance(vector1, vector2, p):

    distance = 0

    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i]) ** p

    return distance ** (1 / p)


# AI Assisted Code
# Function to calculate Mean
def calculate_mean(values):

    return sum(values) / len(values)


# AI Assisted Code
# Function to calculate Variance
def calculate_variance(values):

    mean = calculate_mean(values)

    variance = sum((value - mean) ** 2 for value in values) / len(values)

    return variance


# AI Assisted Code
# Function to calculate Standard Deviation
def calculate_standard_deviation(values):

    return calculate_variance(values) ** 0.5


# Unit Test Class
class TestLabFunctions(unittest.TestCase):

    # Test Mean Function
    def test_mean(self):

        values = [2, 4, 6, 8]

        self.assertEqual(calculate_mean(values), 5)

    # Test Variance Function
    def test_variance(self):

        values = [2, 4, 6, 8]

        self.assertAlmostEqual(calculate_variance(values), 5)

    # Test Standard Deviation Function
    def test_standard_deviation(self):

        values = [2, 4, 6, 8]

        self.assertAlmostEqual(calculate_standard_deviation(values), np.sqrt(5))

    # Test Minkowski Distance
    def test_minkowski_distance(self):

        vector1 = [1, 2]

        vector2 = [4, 6]

        distance = minkowski_distance(vector1, vector2, 2)

        self.assertAlmostEqual(distance, 5.0)


# Main Function
if __name__ == "__main__":

    unittest.main()