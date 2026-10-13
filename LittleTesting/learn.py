def get_positive_numbers(numbers):
    results = []

    for number in numbers:
        if(number > 0):
            results.append(number)

    return results

def calculate_positive_average(numbers):
    total = 0
    count = 0
    average = 0

    for number in numbers:
        if(number > 0):
            sum += number
            count += 1

    if(count > 0):
        average = sum / count

    return average

def find_longest_word(words):
    maxlen = 0
    longestword = ""

    for word in words:
        if(len(word) > maxlen):
            longestword = word
            maxlen = len(word)

    return longestword

def find_largest_number(numbers):
    if not numbers:
        return None
    
    maxnumber = numbers[0]

    for number in numbers:
        if(number > maxnumber):
            maxnumber = number

    return maxnumber

def count_vowels(text):
    vowel_count = 0

    for char in text:
        if char in "aeiou":
            vowel_count += 1

    return vowel_count