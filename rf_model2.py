import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

########################################
# Step 1: Data Loading and Preparation
########################################

# Load the original dataset
df = pd.read_csv("FINAL_USO.csv", parse_dates=["Date"], index_col="Date")

# Remove trend columns
trend_columns = [col for col in df.columns if col.endswith('_Trend')]
df = df.drop(columns=trend_columns, errors='ignore')

# Check if 'Adj Close' exists
if 'Adj Close' not in df.columns:
    raise ValueError("Expected 'Adj Close' column not found in the dataset.")

# Add rolling features for Adj Close
df['Adj_Close_7d'] = df['Adj Close'].rolling(window=7, min_periods=7).mean()
df['Adj_Close_30d'] = df['Adj Close'].rolling(window=30, min_periods=30).mean()

# Drop rows with NaN caused by rolling
df = df.dropna()

########################################
# Step 2: Creating Lagged Features
########################################

# Define base predictors including new rolling features
base_predictors = [
    'Adj Close', 'SP_close', 'DJ_close', 'USDI_Price', 'EU_Price', 'GDX_Close',
    'SF_Price', 'PLT_Price', 'PLD_Price', 'RHO_PRICE', 'USO_Close', 'OF_Price', 'OS_Price',
    'Adj_Close_7d', 'Adj_Close_30d'
]

# Filter to ensure they exist in df
base_predictors = [p for p in base_predictors if p in df.columns]

# Create lagged_df
lagged_df = pd.DataFrame(index=df.index)
lagged_df['Adj Close'] = df['Adj Close']  # current day target

for var in base_predictors:
    lagged_df[var + '_prev'] = df[var].shift(1)

# Drop the first row with missing previous day data
lagged_df.dropna(inplace=True)

########################################
# Step 3: Baseline Evaluation
########################################

target = 'Adj Close'
predictors = [col for col in lagged_df.columns if col.endswith('_prev')]

X = lagged_df[predictors].values
y = lagged_df[target].values

kf = KFold(n_splits=10, shuffle=True, random_state=42)
baseline_predictor = 'Adj Close_prev'
X_baseline = lagged_df[baseline_predictor].values

baseline_rmse_scores = []
for train_index, test_index in kf.split(X_baseline):
    y_test = y[test_index]
    y_pred_baseline = X_baseline[test_index]
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    baseline_rmse_scores.append(rmse)

mean_baseline_rmse = np.mean(baseline_rmse_scores)
print(f"Baseline (Previous Day Adj Close) Mean RMSE: {mean_baseline_rmse:.4f}")

########################################
# Step 4: Random Forest Tuning and Evaluation
########################################

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse_score, greater_is_better=False)

# Parameter space for tuning
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20],
    'max_features': ['None', 'sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

rf = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,  # more iterations to find better params
    scoring=rmse_scorer,
    cv=kf,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X, y)

best_rmse = abs(random_search.best_score_)
best_params = random_search.best_params_

print("\nBest parameters found:", best_params)
print(f"Best RMSE from RandomizedSearchCV (CV estimate): {best_rmse:.4f}")

best_rf = random_search.best_estimator_

# Evaluate tuned RF with 10-fold CV again
rmse_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)

mean_rf_rmse = np.mean(rmse_scores)
std_rf_rmse = np.std(rmse_scores)

print("\nTuned Random Forest 10-Fold CV RMSE Scores:", rmse_scores)
print(f"Tuned Random Forest Mean RMSE: {mean_rf_rmse:.4f} ± {std_rf_rmse:.4f}")

# Compare to baseline
print(f"\nBaseline Mean RMSE: {mean_baseline_rmse:.4f}")
if mean_rf_rmse < mean_baseline_rmse:
    print("The tuned Random Forest model now outperforms the baseline!")
else:
    print("The baseline still outperforms the tuned Random Forest model.")
