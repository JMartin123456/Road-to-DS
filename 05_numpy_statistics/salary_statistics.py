from pathlib import Path
import pandas as pd
import numpy as np

# ==========================
#Load from file
# ==========================
BASE_DIR = Path(__file__).parent

input_file = BASE_DIR / "data" / "employees_clean.csv"

df = pd.read_csv(input_file)

salary = df["Salary"].to_numpy()

def salary_statistics(salary):
    return {
        "min": np.min(salary),
        "max": np.max(salary),
        "mean": np.mean(salary),
        "median": np.median(salary),
        "std": np.std(salary),
        "count": salary.size,
        "variance": np.var(salary)
    }

# ==========================
# Outliers
# ==========================
def find_outliers(salary):

    q1 = np.percentile(salary,25)
    q3 = np.percentile(salary,75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return salary[
        (salary < lower) |
        (salary > upper)
    ]

def remove_outliers(salary):
    q1 = np.percentile(salary, 25)
    q3 = np.percentile(salary, 75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    clean_salary = salary[
        (salary >= lower) &
        (salary <= upper)
    ]

    return clean_salary

def compare_salary(before, after):
    print("Before cleaning:")
    print("Average:", np.mean(before))

    print()

    print("After cleaning:")
    print("Average:", np.mean(after))

print("Salary report")
print("================")

stats = salary_statistics(salary)

for key, value in stats.items():
    print(key, ":", value)


salary_with_outlier = np.append(
    salary,
    50000
)

print("Original outliers:")
print(find_outliers(salary_with_outlier))


clean_salary = remove_outliers(salary_with_outlier)

print("Clean data:")
print(clean_salary)

print("After cleaning:")
print(find_outliers(clean_salary))

compare_salary(
    salary_with_outlier,
    clean_salary
)