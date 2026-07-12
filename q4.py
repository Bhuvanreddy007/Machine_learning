def transpose(matrix, rows, cols):
    print("Transpose Matrix:")

    for i in range(cols):
        for j in range(rows):
            print(matrix[j][i], end=" ")
        print()


def main():
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))

    matrix = []

    print("Enter Matrix:")
    for i in range(rows):
        row = list(map(int, input().split()))
        matrix.append(row)

    transpose(matrix, rows, cols)


main()