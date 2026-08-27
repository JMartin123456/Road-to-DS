# 1

def sum_even(numbers):
    total = 0
    for number in numbers:
        if number % 2 == 0:
            total += number
    return total

list = [1, 2, 3, 4, 5, 6]
print(sum_even(list))


# 2

def reverse_string(text):
    reverse_text = ""
    for char in text:
        reverse_text = char + reverse_text
    print(reverse_text)
    
reverse_me = "abc"
reverse_string(reverse_me)

# 3

def is_palindrom(text):
    text_len = len(text)
    for number in range(len(text)):
        if text[number].lower() != text[text_len - 1].lower():
            return False
        text_len -= 1
    return True    

palindrom_string = "Radar"
not_palindrom_string = "notradar"

print(is_palindrom(palindrom_string))

# 4

def get_adults(users):
    adults = []
    for user in users:
        if user.get("age", 0) >= 18:
            adults.append(user["name"])
    return adults

users = [
    {"name": "Peter", "age": 25},
    {"name": "Jana", "age": 17},
    {"name": "Martin", "age": 31},
    {"name": "Eva", "age": 16},
]

print(get_adults(users))

# 5

def users_to_dict(users):
    result = {}
    for user in users:
        result[user["name"]] = user.get("age", 0)
    return result

users = [
    {"name": "Peter", "age": 25},
    {"name": "Jana", "age": 17},
    {"name": "Martin", "age": 31},
]

print(users_to_dict(users))

# 6

def count_errors(logs):
    total_errors = 0
    for log in logs:
        splitted = log.split()
        if splitted[0] == "ERROR":
            total_errors += 1
    return total_errors

logs = [
    "INFO User logged in",
    "ERROR Database failed",
    "INFO User logged out",
    "WARNING Disk almost full",
    "ERROR Timeout",
    "ERROR Connection refused",
]

print(count_errors(logs))