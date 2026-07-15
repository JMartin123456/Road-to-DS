from pathlib import Path
import pandas as pd

#Load from file
print("Súbor:", __file__)
current_dir = Path(__file__).parent
print("Priečinok:", current_dir)
csv_path = current_dir / "sales.csv"
print("CSV:", csv_path)
df = pd.read_csv(csv_path)

#Some information
print(df.head())
print(df.tail(5))
print(df.columns)
print(df.dtypes)

#Shape information
print("Pocet stlpcov: ",df.shape[0])
print("Pocet riadkov: ",df.shape[1])

#New Column
df["Revenue"] = df["Price"] * df["Quantity"]
print(df["Revenue"])
print("Total revenue: ", df["Revenue"].sum())

#Aggregation
product_revenue = df.groupby("Product")["Revenue"].sum()
print(product_revenue)

#The best product
def best_product(product_revenue):
    print("Best product: ", product_revenue.idxmax())
    print("Revenue: ",product_revenue.max())

best_product(product_revenue)

#Best category
print("Best category:")
print(df.groupby("Category")["Revenue"].sum().idxmax())

#Average revenue per sale:
print("Average revenue per sale:")
print(df["Revenue"].mean())

#Sorted from cheapest 
print("Sorted values: ")
print(df.groupby("Product")["Revenue"].sum().sort_values(ascending=False))