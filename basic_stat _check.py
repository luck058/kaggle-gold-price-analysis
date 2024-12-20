import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("FINAL_USO.csv", parse_dates=["Date"], index_col="Date")

# Drop trend columns for clarity in this initial analysis
trend_columns = [col for col in df.columns if col.endswith('_Trend')]
df_reduced = df.drop(columns=trend_columns)

# Define the variables for current-day analysis
variables = [
    'Adj Close',     # Gold ETF Adjusted Close
    'SP_close',      # S&P 500
    'DJ_close',      # Dow Jones
    'USDI_Price',    # US Dollar Index
    'EU_Price',      # EUR/USD
    'GDX_Close',     # Gold Miners ETF
    'SF_Price',      # Silver
    'PLT_Price',     # Platinum
    'PLD_Price',     # Palladium
    'RHO_PRICE',     # Rhodium
    'USO_Close',     # Oil ETF (WTI)
    'OF_Price',      # Brent Crude
    'OS_Price'       # WTI Crude
]

# Filter the DataFrame to only include these columns (if they exist)
variables = [v for v in variables if v in df_reduced.columns]
analysis_df = df_reduced[variables]

# 1. Descriptive Statistics
print("Descriptive Statistics for Selected Variables:\n")
print(analysis_df.describe().T)  # Transpose for better readability

# Print median values separately if desired
print("\nMedian Values:")
print(analysis_df.median())

# 2. Correlation Matrix
corr_matrix = analysis_df.corr()
print("\nCorrelation Matrix:")
print(corr_matrix.to_string())

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Selected Variables (Current Day)")
plt.show()

# 3. Time-Series Plots
# Plot the main target variable (Gold ETF Adj Close)
plt.figure(figsize=(12, 6))
plt.plot(analysis_df.index, analysis_df['Adj Close'], label='Gold ETF Adj Close', color='gold')
plt.title("Gold ETF Adjusted Close Price Over Time (Current Day Data)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# Plot selected related instruments to observe co-movements
other_vars = [v for v in variables if v != 'Adj Close']

plt.figure(figsize=(12, 6))
for var in other_vars:
    plt.plot(analysis_df.index, analysis_df[var], label=var)
plt.title("Selected Related Markets Over Time (Current Day Data)")
plt.xlabel("Date")
plt.ylabel("Price / Index Level")
plt.legend()
plt.grid(True)
plt.show()