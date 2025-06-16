# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
from utils import make_prediction

app = Flask(__name__)
model = joblib.load('model/random_forest_model.pkl')  # Load your model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })

        prediction = make_prediction(model, input_data)
        return render_template('index.html', prediction=f"₹{prediction:,.2f}")

if __name__ == '__main__':
    app.run(debug=True)
    