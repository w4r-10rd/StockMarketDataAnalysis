import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "complete_dataset.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Ensure data is sorted by date
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
df = df.sort_values(by='Date')

# Debug: Check dataset structure
print("Initial dataset shape:", df.shape)
print(df.head())

# Feature engineering
df['Close_lag1'] = df['Close'].shift(1)
df['Close_lag2'] = df['Close'].shift(2)
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()
df['Volatility'] = df['Close'].rolling(window=7).std()

# Drop rows with NaN values after feature engineering
df = df.dropna(subset=['Close', 'Close_lag1', 'Close_lag2', 'MA_7', 'MA_30', 'Volatility'])
print("Dataset after preprocessing shape:", df.shape)

# Ensure there is sufficient data
if df.empty:
    raise ValueError("Insufficient data after preprocessing. Check your dataset and ensure it has enough rows.")

# Define features and target variable
y = df['Close']
X = df[['Close_lag1', 'Close_lag2', 'MA_7', 'MA_30', 'Volatility']]

# Debug: Check features
print("Features shape:", X.shape)
print(X.head())

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

# Train the XGBoost model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Save the predictions to a CSV file for reference
output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
output.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'.")
