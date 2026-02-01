import pandas as pd
import os

# ================================
# LOAD RAW DATA
# ================================
INPUT_FILE = "fruit_yield_raw.csv"
OUTPUT_DIR = "country_wise_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# ================================
# KEEP ONLY USEFUL COLUMNS
# ================================
df = df[["Area", "Item", "Year", "Element", "Value"]]

# ================================
# TYPE CLEANING
# ================================
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# ================================
# REMOVE MISSING DATA
# ================================
df = df.dropna().reset_index(drop=True)

# ================================
# KEEP REQUIRED MEASURES
# ================================
df = df[df["Element"].isin([
    "Yield",
    "Production",
    "Area harvested"
])]

# ================================
# STRUCTURAL PROCESSING
# (Long → Wide)
# ================================
df = df.pivot_table(
    index=["Area", "Item", "Year"],
    columns="Element",
    values="Value",
    aggfunc="mean"
).reset_index()

# ================================
# RENAME COLUMNS
# ================================
df.rename(columns={
    "Area harvested": "Area_Harvested"
}, inplace=True)

# ================================
# FINAL CLEAN
# ================================
df = df.dropna().reset_index(drop=True)

# ================================
# COUNTRY-WISE PROCESSING
# ================================
for country in df["Area"].unique():
    country_df = df[df["Area"] == country]

    safe_name = country.replace(" ", "_")
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}.csv"
    )

    country_df.to_csv(output_path, index=False)

    print(f"✅ Saved: {output_path}")

print("🎯 CLEANING AND DATA PROCESSING IS COMPLETED")




