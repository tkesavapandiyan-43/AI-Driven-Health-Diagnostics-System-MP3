import joblib
import numpy as np
import pandas as pd

def ValuePredictor(to_predict_list):
    """
    Detects which disease based on input length and returns:
    prediction value and page name
    """

    if len(to_predict_list) == 24:
        page = 'kidney'
        model = joblib.load('./website/app_models/kidney_model.pkl')
        scaler = joblib.load('./website/app_models/kidney_scaler.pkl')
        scaled_input = scaler.transform(np.array(to_predict_list).reshape(1, -1))
        pred = model.predict(scaled_input)

    elif len(to_predict_list) == 10:
        page = 'liver'
        model = joblib.load('./website/app_models/liver_model.pkl')
        scaler = joblib.load('./website/app_models/liver_scaler.pkl')
        scaled_input = scaler.transform(np.array(to_predict_list).reshape(1, -1))
        pred = model.predict(scaled_input)

    elif len(to_predict_list) == 13:
        page = 'heart'
        model = joblib.load('./website/app_models/heart_model.pkl')
        scaler = joblib.load('./website/app_models/heart_scaler.pkl')
        scaled_input = scaler.transform(np.array(to_predict_list).reshape(1, -1))
        pred = model.predict(scaled_input)

    elif len(to_predict_list) == 9:
        page = 'stroke'
        model = joblib.load('./website/app_models/stroke_model.pkl')
        scaler = joblib.load('./website/app_models/stroke_scaler.pkl')
        scaled_input = scaler.transform(np.array(to_predict_list).reshape(1, -1))
        pred = model.predict(scaled_input)

    elif len(to_predict_list) == 8:
        page = 'diabetes'
        model = joblib.load('./website/app_models/diabetes_model.pkl')
        scaler = joblib.load('./website/app_models/diabetes_scaler.pkl')
        feature_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                         'Insulin','BMI','DiabetesPedigreeFunction','Age']
        df_input = pd.DataFrame([to_predict_list], columns=feature_names)
        scaled_input = scaler.transform(df_input)
        pred = model.predict(df_input)

    else:
        raise ValueError("Input list length does not match any supported disease model.")

    return pred[0], page
