def count(text):
    vowel = 0
    consonant = 0

    for ch in text.lower():
        if ch.isalpha():
            if ch in "aeiou":
                vowel += 1
            else:
                consonant += 1

    print("Vowels:", vowel)
    print("Consonants:", consonant)


def main():
    text = input("Enter a string: ")
    count(text)


main()