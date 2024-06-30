import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# Load the dataset
url = "car+data.csv"
data = pd.read_csv(url)

# Check the shape
print(data.shape)

# Check the basic information
print(data.info())

# Display the first few rows of the dataset
print(data.head())

# Check for duplicates
duplicates = data.duplicated().sum()
print(f"Number of duplicate records: {duplicates}")

# Drop duplicates if any
if duplicates > 0:
    data = data.drop_duplicates()

# Drop the 'Car_Name' column as it is redundant for the analysis
data = data.drop(['Car_Name'], axis=1)

# Extract 'age_of_the_car' from 'Year'
data['age_of_the_car'] = 2024 - data['Year']  # Assuming the current year is 2024
data = data.drop(['Year'], axis=1)

# Encode categorical columns
data = pd.get_dummies(data, drop_first=True)


# Separate the target and independent features
X = data.drop(['Selling_Price'], axis=1)
y = data['Selling_Price']


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the train and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate the r2-score for train and test sets
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"R2 Score for Train Set: {r2_train}")
print(f"R2 Score for Test Set: {r2_test}")

# Save the model as a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

