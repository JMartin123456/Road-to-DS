from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def clean_salary_outliers(df):

    q1 = np.percentile(df["Salary"], 25)
    q3 = np.percentile(df["Salary"], 75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    clean_df = df[
        (df["Salary"] >= lower) &
        (df["Salary"] <= upper)
    ]

    outliers = df[
        (df["Salary"] < lower) |
        (df["Salary"] > upper)
    ]

    print(outliers)

    return clean_df

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

def create_scatter_plot(df, x_column, y_column, title):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df[x_column],
        df[y_column]
    )

    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(title)

    plt.grid(True)

    plt.show()
    plt.close()

def create_bar_plot(df, x_column):
    plt.figure(figsize=(8, 5))

    counts = df[x_column].value_counts()
    plt.bar(counts.index, counts.values)

    plt.xlabel(x_column)
    plt.ylabel("Count")
    plt.title(f"{x_column} Distribution")

    plt.grid(axis="y", alpha=0.3)

    plt.show()
    plt.close()

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

# 3) Education vs Salary, Does employees with higher education earn more ?

salary_by_education_mean = df.groupby("Education")["Salary"].mean()
salary_by_education_median = df.groupby("Education")["Salary"].median()
salary_by_education_size = df.groupby("Education")["Salary"].size()
salary_by_education_std = df.groupby("Education")["Salary"].std()

salary_vs_education = pd.DataFrame({
    "Mean": salary_by_education_mean,
    "Median": salary_by_education_median,
    "Size": salary_by_education_size,
    "Std": salary_by_education_std
})

salary_vs_education["Difference"] = salary_vs_education["Mean"] - salary_vs_education["Median"]

print()
print("Salary vs education")
print(salary_vs_education)

print()
print("Education Master")
print(df.loc[
    df["Education"] == "Master",
    ["Name", "Department", "Salary", "Experience"]
].sort_values("Salary")
)

# Education alone does not show a clear linear relationship with salary. 
# The results are significantly influenced by extreme values.

# 4) PerformanceScore vs Salary. Does people with better skore earn more?


df_without_salary_outliers = clean_salary_outliers(df)

print(df.shape)
print(df_without_salary_outliers.shape)

df_without_salary_outliers["PerfromanceScoreGroup"] = pd.cut(
    df["PerformanceScore"],
    bins=[2.5, 3, 3.5, 4, 4.5, 5],
    labels=[
        "2.5 to 3",
        "3 to 3.5",
        "3.5 to 4",
        "4 to 4.5",
        "4.5+"
    ]
)

salary_by_performance_mean = df_without_salary_outliers.groupby("PerfromanceScoreGroup")["Salary"].mean()
salary_by_performance_median = df_without_salary_outliers.groupby("PerfromanceScoreGroup")["Salary"].median()
salary_by_performance_size = df_without_salary_outliers.groupby("PerfromanceScoreGroup")["Salary"].size()
salary_by_performance_std = df_without_salary_outliers.groupby("PerfromanceScoreGroup")["Salary"].std()

salary_vs_performance = pd.DataFrame({
    "Mean": salary_by_performance_mean,
    "Median": salary_by_performance_median,
    "Size": salary_by_performance_size,
    "Std": salary_by_performance_std
})

salary_vs_performance["Difference"] = salary_vs_performance["Mean"] - salary_vs_performance["Median"]

print()
print("Salary vs performance")
print(salary_vs_performance)

# After removing extreme salary observations, no clear relationship between PerformanceScore and Salary was identified.

correlation = df.corr(numeric_only=True)

print(correlation)

# ==========================
# Graphs
# ==========================

# Experience vs Salary
# original
create_scatter_plot(df, "Experience", "Salary", "Experience vs Salary")

# clean
create_scatter_plot(df_without_salary_outliers, "Experience", "Salary", "Experience vs Salary")

# Performance vs Salary 
# original
create_scatter_plot(df, "PerformanceScore", "Salary", "PerfromanceScore vs Salary")

# clean
create_scatter_plot(df_without_salary_outliers, "PerformanceScore", "Salary", "PerfromanceScore vs Salary")

# Age vs Salary 
# original
create_scatter_plot(df, "Age", "Salary", "Age vs Salary")

# clean
create_scatter_plot(df_without_salary_outliers, "Age", "Salary", "Age vs Salary")

correlation_clean = df_without_salary_outliers.corr(numeric_only=True)
print(correlation_clean)

create_bar_plot(df, "Department")
create_bar_plot(df, "Education")

plt.figure(figsize=(6, 5))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()
plt.close()