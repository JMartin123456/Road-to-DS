from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
#Load from file
# ==========================
BASE_DIR = Path(__file__).parent

IMAGE_DIR = BASE_DIR / "images"

IMAGE_DIR.mkdir(exist_ok=True)

input_file = BASE_DIR / "data" / "employees_clean.csv"

df = pd.read_csv(input_file)

print(df.head())

# ==========================
#number of employees by department
# ==========================
department_count = df.groupby("Department").size()
#same as
#department_count = df["Department"].value_counts()
print(department_count)

plt.title("Number of employees by department")
plt.bar(department_count.index, department_count.values)
plt.xlabel("Department name")
plt.ylabel("Employee count")
plt.savefig(
    IMAGE_DIR / "employees_by_department.png"
)
plt.close()

# ==========================
# Average Salary by Department
# ==========================
average_salary = df.groupby("Department")["Salary"].mean()
plt.figure(figsize=(8,5))

plt.title("Average Salary by Department")

plt.bar(
    average_salary.index,
    average_salary.values
)

plt.xlabel("Department")
plt.ylabel("Average Salary")

plt.grid(axis="y")

plt.savefig(
    IMAGE_DIR / "average_salary_by_department.png"
)
plt.close()
# ==========================
# Histogram age
# ==========================
plt.figure(figsize=(8, 5))

plt.title("Age Histogram")

bins = 10
plt.hist(
    df["Age"],
    bins
)

plt.xlabel("Age")
plt.ylabel("People count of that age")

plt.savefig(
    IMAGE_DIR / "age_histogram.png"
)
plt.close()

# ==========================
# Age vs Salary
# ==========================
plt.figure(figsize=(8, 5))

plt.title("Age vs Salary")

bins = 10
plt.scatter(
    df["Salary"],
    df["Age"],
    alpha=0.7
)

plt.xlabel("Salary")
plt.ylabel("Age")

plt.grid(axis="y")
plt.grid(axis="x")

plt.savefig(
    IMAGE_DIR / "age_vs_salary.png"
)
plt.close()

# ==========================
# Employees by City
# ==========================
city_count = df.groupby("City").size()
plt.figure(figsize=(8,5))

plt.title("Employees by City")

plt.bar(
    city_count.index,
    city_count.values
)

plt.xlabel("City")
plt.ylabel("People count")

plt.savefig(
    IMAGE_DIR / "employees_by_city.png"
)
plt.close()

# ==========================
# Employees by city, pie edition
# ==========================
plt.pie(
    city_count.values,
    labels= city_count.index 
)

plt.savefig(
    IMAGE_DIR / "employees_by_city_pie_chart.png"
)
plt.close()
