def input_matrix(rows, cols):
    matrix = []

    for i in range(rows):
        row = list(map(int, input().split()))
        matrix.append(row)

    return matrix


def multiply(mat1, mat2, rows1, cols1, cols2):
    for i in range(rows1):
        for j in range(cols2):
            total = 0
            for k in range(cols1):
                total += mat1[i][k] * mat2[k][j]
            print(total, end=" ")
        print()


def main():
    rows1 = int(input("Enter rows of Matrix A: "))
    cols1 = int(input("Enter columns of Matrix A: "))
    print("Enter Matrix A:")
    mat1 = input_matrix(rows1, cols1)

    rows2 = int(input("Enter rows of Matrix B: "))
    cols2 = int(input("Enter columns of Matrix B: "))
    print("Enter Matrix B:")
    mat2 = input_matrix(rows2, cols2)

    if cols1 != rows2:
        print("Matrix multiplication is not possible.")
    else:
        print("Product Matrix:")
        multiply(mat1, mat2, rows1, cols1, cols2)


main()