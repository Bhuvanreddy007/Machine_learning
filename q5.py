import random
import statistics


def calculate(numbers):
    print("Generated Numbers:")
    print(numbers)

    print("Mean:", statistics.mean(numbers))
    print("Median:", statistics.median(numbers))
    print("Mode:", statistics.mode(numbers))


def main():
    start = int(input("Enter starting number: "))
    end = int(input("Enter ending number: "))

    numbers = []

    for i in range(100):
        numbers.append(random.randint(start, end))

    calculate(numbers)


main()