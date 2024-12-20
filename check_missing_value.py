import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("FINAL_USO.csv", parse_dates=["Date"], index_col="Date")

# 1. Check for Missing Values (NaN)
missing_counts = df.isna().sum()
print("Missing values per column:\n", missing_counts.to_string())

# For total missing values, just print the integer directly
total_missing = df.isna().sum().sum()
print("\nTotal number of missing values:", total_missing)

# 2. Check Data Types
print("\nData types of each column:")
print(df.dtypes.to_string())


