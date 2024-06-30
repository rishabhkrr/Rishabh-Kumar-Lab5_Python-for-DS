import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
url = "car+data.csv"
data = pd.read_csv(url)

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

# Save the model as a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)