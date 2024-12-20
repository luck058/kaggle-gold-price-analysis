import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("FINAL_USO.csv", parse_dates=["Date"], index_col="Date")

# 2. Check the shape of the dataset
print("Number of rows (days):", df.shape[0])
print("Number of columns (variables):", df.shape[1])

# 3. Display the first few rows to understand the data structure
print(df.head())

# 4. Check if the Date index is continuous or if there are gaps (holidays/weekends)
# Since this is financial data (ETF data from Yahoo Finance), typically weekends and holidays are not included.
# We can verify by checking the day of the week distribution.
print("Unique days of week in data:", df.index.dayofweek.unique())

# dayofweek: Monday=0, Tuesday=1, ... Friday=4
# If we see no 5 or 6, likely no weekends are included.

# 5. Check for missing dates:
# Create a range of dates from start to end and compare to the dataset index.
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B') # 'B' = business days
missing_dates = full_range.difference(df.index)
print("Number of missing business days (potential holidays):", len(missing_dates))
if len(missing_dates) > 0:
    print("Missing dates:", missing_dates)

# From these checks:
# - We know exactly how many observations and variables are in the dataset.
# - By examining dayofweek, we can confirm weekends are excluded (common for financial data).
# - By checking missing business days, we can see if some holidays or non-trading days are absent from the data.
