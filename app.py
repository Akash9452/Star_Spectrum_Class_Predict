from flask import Flask, render_template, request, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('Star_Class_type.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess and predict
def predict_star_class(features):
    input_features = [
        features['Temperature'], 
        features['Luminosity'], 
        features['Radius'], 
        features['Absolute_magnitude']
    ]

    # Add the star type mapped value
    input_features.append(features['Star_type'])

    # Add dummy variables for Star_Color
    input_features.extend(features['Star_Color'])

    input_features = np.array([input_features])
    prediction = model.predict(input_features)
    return prediction[0]  # Assuming the model's predict method returns a list/array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temperature = request.form.get('Temperature')
    luminosity = request.form.get('Luminosity')
    radius = request.form.get('Radius')
    absolute_magnitude = request.form.get('Absolute_magnitude')
    star_type = request.form.get('Star_type')
    star_color = request.form.get('Star_Color')

    # Mapping for star type
    star_type_mapping = {
        "Brown Dwarf": 0,
        "Red Dwarf": 1,
        "White Dwarf": 2,
        "Main Sequence": 3,
        "SuperGiants": 4,
        "HyperGiants": 5
    }

    # Dummy variable mapping for star color
    star_color_options = ['Red', 'Blue White', 'White', 'Yellowish White', 'Blue white',
       'Pale yellow orange', 'Blue', 'Blue-white', 'Whitish',
       'yellow-white', 'Orange', 'White-Yellow', 'white', 'Blue ',
       'yellowish', 'Yellowish', 'Orange-Red', 'Blue white ',
       'Blue-White']
    star_color_dummies = [1 if color == star_color else 0 for color in star_color_options]

    features = {
        'Temperature': float(temperature),
        'Luminosity': float(luminosity),
        'Radius': float(radius),
        'Absolute_magnitude': float(absolute_magnitude),
        'Star_type': star_type_mapping.get(star_type),
        'Star_Color': star_color_dummies
    }

    prediction = predict_star_class(features)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
