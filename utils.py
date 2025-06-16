# utils.py
import numpy as np
import pandas as pd

def preprocess_input(data):
    df = data.copy()
    df['age*bmi'] = df['age'] * df['bmi']
    df['is_obese'] = (df['bmi'] > 30).astype(int)
    df['age*smoker'] = df['age'] * (df['smoker'] == 'yes').astype(int)
    return df

def make_prediction(model, input_data):
    processed = preprocess_input(input_data)
    pred_log = model.predict(processed)[0]
    return np.exp(pred_log) - 1
