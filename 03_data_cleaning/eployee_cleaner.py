from pathlib import Path
import pandas as pd

#Load from file
print("Súbor:", __file__)
current_dir = Path(__file__).parent
print("Priečinok:", current_dir)
csv_path = current_dir / "employees.csv"
print("CSV:", csv_path)
df = pd.read_csv(csv_path)

#info
print(df.head())
print(df.info())
print(df.describe())

#what is wrong
print("Information-------------------------------------------")
print(df.isna().sum())
print(df.duplicated())
print(df[df.duplicated])
print(df["Department"].unique())
print(df.dtypes)
print(df["Age"])

#Additional info
print("Part 3-------------------------------------------")
print(df["Age"].unique())
print(df["Salary"].unique())

#Start of cleaning
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
print(df.dtypes)
print(df["Age"].isna().sum())

#dropping Naan values OR fill with median
#This is Dropping option: 
#df = df.dropna(subset=["Age"])
print("Age median: " ,df["Age"].median())
df["Age"] = df["Age"].fillna(df["Age"].median())
print(df["Age"].isna().sum())

#resolving duplicity
print("")
df.duplicated()
print(df.isna().sum())
print(df.duplicated())