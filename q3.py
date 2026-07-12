def common(list1, list2):
    result = []

    for num in list1:
        if num in list2 and num not in result:
            result.append(num)

    print("Common Elements:", result)
    print("Count:", len(result))


def main():
    list1 = list(map(int, input("Enter first list: ").split()))
    list2 = list(map(int, input("Enter second list: ").split()))

    common(list1, list2)


main()