customers = [
    {
        "name": "Anna",
        "age": 25,
        "spent": 1200,
        "city": "Kosice"
    },
    {
        "name": "Peter",
        "age": 35,
        "spent": 3000,
        "city": "Kosice"
    },
    {
        "name": "Eva",
        "age": 29,
        "spent": 800,
        "city": "Kosice"
    },
    {
        "name": "Martin",
        "age": 42,
        "spent": 5500,
        "city": "Bratislava"
    }
]

def calculate_total_sales(customers):
    total_sales = 0
    for customer in customers:
        total_sales += customer["spent"]
    return total_sales



def find_best_customer(customers):
    
    if not customers:
        return None
    
    best_customer = None
    highest_spending = 0
    for customer in customers:
        if customer["spent"] > highest_spending:
            highest_spending = customer["spent"]
            best_customer = customer
    return best_customer


def average_spending(customers):

    if not customers:
        return 0

    total = 0
    for customer in customers:
        total += customer["spent"]
    return round(total / len(customers), 2)


def classify_customer(customer):
    spent = customer["spent"]

    if spent > 2000:
        return  "VIP"
    elif spent > 1000:
        return "Regular"
    else:
        return "Low"
    
def customers_from_city(customers, city):

    if not customers:
        return []
    
    result = []
    for customer in customers:
        if customer["city"] == city:
            result.append(customer)
    return result


def top_customers(customers, limit):

    if not customers:
        return []
    
    result_list = sorted(
        customers,
        key=lambda customer: customer["spent"],
        reverse=True
    )

    return result_list[:limit]
    


if __name__ == "__main__":

    print(calculate_total_sales(customers))
    print(find_best_customer(customers))
    print(average_spending(customers))

    for customer in customers:
        print(
            customer["name"],
            "->",
            classify_customer(customer)
        )

    print(customers_from_city(customers, "Bratislava"))
    print(top_customers(customers, 2))