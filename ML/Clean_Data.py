import pandas as pd

# Load raw data
df = pd.read_csv("fruit_yield.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Clean string columns (Area, Item, etc.)
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip().str.replace('"', '')

# Convert numeric columns
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# Remove rows with missing Year or Value
df_clean = df.dropna(subset=["Year", "Value"])

# Sort by Year (optional but recommended)
df_clean = df_clean.sort_values("Year").reset_index(drop=True)

# Save FULL cleaned table
df_clean.to_csv("fruit_yield_cleaned.csv", index=False)

print("✅ Full cleaned table saved as fruit_yield_cleaned.csv")
print(df_clean.head())


