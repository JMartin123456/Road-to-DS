# Employee Data Cleaning Project

## Goal

Take a raw employee dataset containing missing values, incorrect data types and duplicate record and clean the dataset.

## Technologies Used
-Python 3
-Pandas
-pathlib

## Dataset

The dataset contains employee information:

- Name
- Age
- Department
- Salary
- City

## Data Cleaning
Invalid values
pd.to_numeric(errors="coerce")

Missing values
groupby("Department").transform("mean")

And remove duplicates
df.drop_duplicates()

