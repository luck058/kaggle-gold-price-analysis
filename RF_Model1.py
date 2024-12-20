import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

# Load the original dataset
df = pd.read_csv("FINAL_USO.csv", parse_dates=["Date"], index_col="Date")

# Drop trend columns to simplify
trend_columns = [col for col in df.columns if col.endswith('_Trend')]
df_reduced = df.drop(columns=trend_columns)

# Define the target and predictor variables
target_column = 'Adj Close'  # Current day’s Gold ETF Adjusted Close as target
predictor_vars = [
    'Adj Close',    # Include the Gold ETF's own Adj Close for the previous day
    'SP_close',     # S&P 500
    'DJ_close',     # Dow Jones
    'USDI_Price',   # US Dollar Index
    'EU_Price',     # EUR/USD
    'GDX_Close',    # Gold Miners ETF
    'SF_Price',     # Silver
    'PLT_Price',    # Platinum
    'PLD_Price',    # Palladium
    'RHO_PRICE',    # Rhodium
    'USO_Close',    # Oil ETF (WTI)
    'OF_Price',     # Brent Crude
    'OS_Price'      # WTI Crude
]

# Ensure all selected columns exist
all_vars = [v for v in predictor_vars if v in df_reduced.columns]
if target_column not in all_vars:
    all_vars = [target_column] + all_vars
analysis_df = df_reduced[all_vars].copy()

# Create a new DataFrame with the current day target
lagged_df = pd.DataFrame(index=analysis_df.index)
lagged_df[target_column] = analysis_df[target_column]  # Current day target

# Shift the predictor variables by 1 to represent the previous day's values
for var in predictor_vars:
    if var in analysis_df.columns:
        lagged_df[var + "_prev"] = analysis_df[var].shift(1)

# Drop the first row because it does not have a previous day available
lagged_df.dropna(inplace=True)

print(lagged_df.head())

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Create an RMSE scorer where lower is better
rmse_scorer = make_scorer(rmse_score, greater_is_better=False)

target = 'Adj Close'
predictors = [col for col in lagged_df.columns if col.endswith('_prev')]

X = lagged_df[predictors].values
y = lagged_df[target].values

# Define parameter space for Random Forest hyperparameter tuning
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Set up 10-fold CV and RandomizedSearchCV
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
rf = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,  # Try 20 random parameter combinations
    scoring=rmse_scorer,
    cv=kfold,
    random_state=42,
    n_jobs=-1
)

# Fit the RandomizedSearchCV to find the best hyperparameters
random_search.fit(X, y)

# random_search.best_score_ is negative RMSE, so we take its absolute value
best_rmse = abs(random_search.best_score_)
best_params = random_search.best_params_

print("Best parameters found:", best_params)
print(f"Best RMSE from RandomizedSearchCV: {best_rmse:.4f}")

# Retrain a RandomForest with the best found parameters
best_rf = random_search.best_estimator_

# Evaluate using 10-fold CV again to confirm performance
rmse_scores = []
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)

mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

print("\nTuned Random Forest 10-Fold CV RMSE Scores:", rmse_scores)
print(f"Tuned Random Forest Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

# Provide a summary interpretation
print("\nSummary:")
print("The RandomizedSearchCV identified optimal parameters that yield a best RMSE "
      f"of approximately {best_rmse:.4f} during tuning.")
print("After refitting with these parameters and re-evaluating with 10-fold CV, "
      f"the model achieves a mean RMSE of about {mean_rmse:.4f}. "
      "If this is an improvement over the previous model or baseline, it indicates that "
      "hyperparameter tuning helped reduce error.")

target = 'Adj Close'
predictors = [col for col in lagged_df.columns if col.endswith('_prev')]

X = lagged_df[predictors].values
y = lagged_df[target].values

# 10-Fold Cross Validation Setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Random Forest Evaluation
rmse_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)

mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

print("Random Forest 10-Fold CV RMSE Scores:", rmse_scores)
print(f"Random Forest Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

# Baseline Model Evaluation
# Baseline: predict today's Adj Close = yesterday's Adj Close (Adj Close_prev)
baseline_predictor = 'Adj Close_prev'
X_baseline = lagged_df[baseline_predictor].values

baseline_rmse_scores = []

for train_index, test_index in kf.split(X_baseline):
    y_test = y[test_index]
    # Baseline prediction is simply the previous day's Adj Close
    y_pred_baseline = X_baseline[test_index]
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    baseline_rmse_scores.append(baseline_rmse)

mean_baseline_rmse = np.mean(baseline_rmse_scores)
std_baseline_rmse = np.std(baseline_rmse_scores)

print("Baseline (Previous Day Adj Close) 10-Fold CV RMSE Scores:", baseline_rmse_scores)
print(f"Baseline Mean RMSE: {mean_baseline_rmse:.4f} ± {std_baseline_rmse:.4f}")

# Comparison
if mean_rmse < mean_baseline_rmse:
    print("The Random Forest model outperforms the baseline.")
else:
    print("The Random Forest model does not outperform the baseline.")