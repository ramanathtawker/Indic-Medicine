import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv('your_data.csv')  # Replace with your actual file path

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Feature engineering
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['week_of_year'] = df['date'].dt.isocalendar().week

# Create lag features
df['prev_day_credit_debit'] = df['credit/debit'].shift(1)
df['prev_week_credit_debit'] = df['credit/debit'].shift(7)

# Create rolling statistics
df['rolling_7d_mean'] = df['credit/debit'].rolling(window=7).mean()
df['rolling_30d_mean'] = df['credit/debit'].rolling(window=30).mean()

# Drop rows with NaN values (from shifting and rolling operations)
df = df.dropna()

# Prepare features and target
features = ['month', 'day_of_week', 'day_of_month', 'week_of_year', 
            'prev_day_credit_debit', 'prev_week_credit_debit', 
            'rolling_7d_mean', 'rolling_30d_mean']

X = df[features]
y = df['credit/debit']

# Encode categorical variables
le = LabelEncoder()
X['month'] = le.fit_transform(X['month'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

# Predict for the upcoming year
# Assuming you have a dataframe 'future_df' with the same features for the upcoming year
# future_predictions = model.predict(future_df[features])
