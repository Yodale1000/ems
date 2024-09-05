import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
import os

# Data Loading and Preparation
df = pd.read_csv('data.csv', sep=';')
df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
df['y'] = df['y'].abs()

# Feature Engineering
df['year'] = df['ds'].dt.year
df['month'] = df['ds'].dt.month
df['day'] = df['ds'].dt.day
df['dayofweek'] = df['ds'].dt.dayofweek

X = df[['year', 'month', 'day', 'dayofweek']]
y = df['y']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# XGBoost Model and Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror'),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=3,
                           verbose=1)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predictions on Test Set
y_pred = best_model.predict(X_test)

# Performance Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'XGBoost Mean Squared Error: {mse}')
print(f'XGBoost Mean Absolute Error: {mae}')
print(f'XGBoost R² Score: {r2}')

# Plotting Future Predictions
periods = 2
future_dates = pd.date_range(start=df['ds'].iloc[-1], periods=periods + 1)
future_features = pd.DataFrame({
    'year': future_dates.year,
    'month': future_dates.month,
    'day': future_dates.day,
    'dayofweek': future_dates.dayofweek
})

future_pred = best_model.predict(future_features)

plt.figure(figsize=(10, 6))
plt.plot(future_dates, future_pred, label='XGBoost Forecast')
plt.xlabel('Date')
plt.ylabel('Y')
plt.title('XGBoost Forecast')
plt.legend()

# Save and Show Plot
os.makedirs('plots/xgboost', exist_ok=True)
plt.savefig('plots/xgboost/forecast.png')
plt.show()
