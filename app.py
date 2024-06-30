from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    present_price = float(request.form['present_price'])
    kms_driven = int(request.form['kms_driven'])
    fuel_type = request.form['fuel_type']
    seller_type = request.form['seller_type']
    transmission = request.form['transmission']
    owner = int(request.form['owner'])
    age_of_the_car = int(request.form['age_of_the_car'])

    # Print the inputs for debugging
    print("Inputs received:")
    print(f"Present Price: {present_price}")
    print(f"Kms Driven: {kms_driven}")
    print(f"Fuel Type: {fuel_type}")
    print(f"Seller Type: {seller_type}")
    print(f"Transmission: {transmission}")
    print(f"Owner: {owner}")
    print(f"Age of the Car: {age_of_the_car}")

    # Encode categorical variables
    fuel_type_Petrol = 0
    fuel_type_Diesel = 0
    if fuel_type == 'Petrol':
        fuel_type_Petrol = 1
    elif fuel_type == 'Diesel':
        fuel_type_Diesel = 1

    seller_type_Individual = 0
    if seller_type == 'Individual':
        seller_type_Individual = 1

    transmission_Manual = 0
    if transmission == 'Manual':
        transmission_Manual = 1

    # Create the input array
    input_features = [present_price, kms_driven, owner, age_of_the_car, fuel_type_Diesel, fuel_type_Petrol,
                      seller_type_Individual, transmission_Manual]
    input_array = np.array([input_features])

    # Print the input array for debugging
    print("Input array for prediction:")
    print(input_array)

    # Predict the price
    prediction = model.predict(input_array)

    # Print the prediction for debugging
    print("Prediction made:")
    print(prediction)

    return render_template('index.html', prediction_text='Estimated Selling Price: ₹ {:.2f} lakhs'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
