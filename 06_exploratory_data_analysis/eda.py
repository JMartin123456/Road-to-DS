from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
#Load from file
# ==========================
BASE_DIR = Path(__file__).parent

input_file = BASE_DIR / "data" / "employees.csv"

df = pd.read_csv(input_file)

def clean_missing_values(df):
    # Data Cleaning
    df["City"] = df["City"].fillna("Unknown")
    df["PerformanceScore"] = df["PerformanceScore"].fillna(df["PerformanceScore"].median())
    df["Education"] =  df["Education"].fillna("Unknown")

    return df

def data_quality_check(df):
    # Data Quality Check
    print("Shape: ", df.shape)
    print("Columns: ", df.columns)
    print("DTypes: ", df.dtypes)
    print("Isna: ", df.isna().sum())
    print("Duplicates: ", df.duplicated().sum())
    print()

def clean_duplicates(df):
    df = df.drop_duplicates()

    return df

def data_validation(df):
    # Data Validation
    print(df.describe())

    categorical_columns = [
        "Gender",
        "Department",
        "City",
        "Education",
        "Remote"
    ]

    #categorical_columns = df.select_dtypes(include="object").columns

    for column in categorical_columns:
        print("--------------------")
        print(df[column].value_counts())

def date_repair(df):

    print(df["HireDate"].str.len().unique())

    converted_dates = pd.to_datetime(
        df["HireDate"],
        errors="coerce"
    )

    print(converted_dates.isna().sum())

    df["HireDate"] = pd.to_datetime(df["HireDate"])
    print(df.dtypes)

    return df

# ==========================
# Data quality check and data cleaning
# ==========================

data_quality_check(df)

df = clean_missing_values(df)

# Check duplicates
duplicates = df[df.duplicated(keep=False)]
print(duplicates)

df = clean_duplicates(df)
data_quality_check(df)

# lets check extreeeeemes
print("Extreeeemees")
data_validation(df)

df = date_repair(df)

print()
data_quality_check(df)

# ==========================
# Exploratory Data Analysis
# ==========================

# Employees
print()
print(df.groupby(["Department"]).size())
print(df.groupby(["Gender"]).size())
print(df.groupby(["City"]).size())
print(df.groupby(["Remote"]).size())

# Salary
print()
print(df["Salary"].mean())
print(df["Salary"].median())
print(df.groupby(["Department"])["Salary"].mean().values)
print(df.groupby(["Department"])["Salary"].median())
print(df.groupby(["Education"])["Salary"].mean())
print(df.groupby(["Education"])["Salary"].median())

# 1) why IT has the highest salary
print(df.groupby(["Department"])["Salary"].max())
print(df.sort_values("Salary", ascending = False)[["Name", "Department", "Salary", "Experience"]].head(10))

mean_by_department = df.groupby(["Department"])["Salary"].mean()
medain_by_department = df.groupby(["Department"])["Salary"].median()


mean_vs_median = pd.DataFrame({
    "Mean": mean_by_department,
    "Median": medain_by_department
})

mean_vs_median["Difference"] = mean_vs_median["Mean"] - mean_vs_median["Median"]
print(mean_vs_median)

# IT has the highest average salary, but the difference is heavily influenced by extreme values. 
# Operations and IT have the largest difference between the mean and median.

# 2) experience vs salary

df["ExperienceGroup"] = pd.cut(
    df["Experience"],
    bins=[-1, 5, 10, 20, 100],
    labels=[
        "0-5 years",
        "6-10 years",
        "11-20 years",
        "20+ years"
    ]
)
print(df[["ExperienceGroup"]])

salary_by_experience_mean = df.groupby("ExperienceGroup")["Salary"].mean()
salary_by_experience_median = df.groupby("ExperienceGroup")["Salary"].median()
salary_by_experience_count = df.groupby("ExperienceGroup")["Salary"].count()

salary_vs_experience = pd.DataFrame({
    "Mean": salary_by_experience_mean,
    "Median": salary_by_experience_median,
    "Count": salary_by_experience_count
})

print(df.loc[
    df["ExperienceGroup"] == "0-5 years",
    ["Name", "Department", "Salary", "Experience"]
])

print(salary_by_experience_mean)
print(salary_by_experience_median)
print(salary_vs_experience)

print(df.loc[
    df["ExperienceGroup"] == "6-10 years",
    ["Name", "Department", "Salary", "Experience"]
].sort_values("Salary")
)

print(df.groupby(["Department", "ExperienceGroup"]).size().unstack(level=-1))

# The group of employees with 6–10 years of experience has the lowest average salary. 
# Detailed analysis showed that this group does not include empoyees from the IT and Operations, which are the best paid.