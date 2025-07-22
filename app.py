from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('logistic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputs = [float(x) for x in request.form.values()]
    scaled_inputs = scaler.transform([inputs])
    prediction = model.predict(scaled_inputs)[0]
    return render_template('index.html', prediction_text=f'Prediction: {"DEATH" if prediction == 1 else "SAFE"}')

if __name__ == '__main__':
    app.run(debug=True)