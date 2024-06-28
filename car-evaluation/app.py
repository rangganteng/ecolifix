from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Define the path to the model
model_filename = os.path.join(os.getcwd(), 'models', 'best_bagging_model.pkl')

# Load the model
try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found: {model_filename}")

# Function to preprocess input data
def preprocess_input(feature1, feature2, feature3, feature4, feature5, feature6, feature7):
    return np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7]])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    feature3 = float(request.form['feature3'])
    feature4 = float(request.form['feature4'])
    feature5 = float(request.form['feature5'])
    feature6 = float(request.form['feature6'])
    feature7 = float(request.form['feature7'])

    # Preprocess the input
    input_data = preprocess_input(feature1, feature2, feature3, feature4, feature5, feature6, feature7)

    # Perform prediction using the model
    prediction = model.predict(input_data)

    # Render the predict.html template with the prediction result
    return render_template('predict.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
