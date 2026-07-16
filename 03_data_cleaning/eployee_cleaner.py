from pathlib import Path
import pandas as pd

#Load from file
BASE_DIR = Path(__file__).parent

input_file = BASE_DIR / "data" / "employees.csv"
output_file = BASE_DIR / "data" / "employees_clean.csv"

df = pd.read_csv(input_file)

#info
# ==========================
# print(df.head())
# print(df.info())
# print(df.describe())

# #what is wrong
# print("Information-------------------------------------------")
# print(df.isna().sum())
# print(df.duplicated())
# print(df[df.duplicated])
# print(df["Department"].unique())
# print(df.dtypes)
# print(df["Age"])

# #Additional info
# print("Part 3-------------------------------------------")
# print(df["Age"].unique())
# print(df["Salary"].unique())
# ==========================

#Start of cleaning
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
#print(df.dtypes)
#print(df["Age"].isna().sum())

#dropping Naan values OR fill with median
#print("Age median: " ,df["Age"].median())
df["Age"] = df["Age"].fillna(df["Age"].median())
#print(df["Age"].isna().sum())

#Salary naan value resolution
#trying to fill with median only from his/her Department
#FAiled df["Salary"] = df["Salary"].fillna(df.groupby("Department")["Salary"].mean())
df["Salary"] = df["Salary"].fillna(df.groupby("Department")["Salary"].transform("mean"))

#City problems and duplicates
df["City"] = df["City"].fillna("Unknown")
df = df.drop_duplicates()


# Save cleaned data
df.to_csv(output_file, index=False)

print("Cleaning finished")
print(f"Saved to: {output_file}")