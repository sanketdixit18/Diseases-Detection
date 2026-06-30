import joblib
import numpy as np
import pandas as pd

# ==========================================
# LOAD MODELS AND SCALERS
# ==========================================

diabetes_model = joblib.load("models/diabetes_model.pkl")
heart_model = joblib.load("models/heart_model.pkl")
breast_model = joblib.load("models/breast_model.pkl")

diabetes_scaler = joblib.load("scalers/diabetes_scaler.pkl")
heart_scaler = joblib.load("scalers/heart_scaler.pkl")
breast_scaler = joblib.load("scalers/breast_scaler.pkl")

# ==========================================
# DIABETES PREDICTION
# ==========================================

def diabetes_prediction():

    print("\nENTER DIABETES DETAILS\n")

    pregnancies = float(input("Pregnancies: "))
    glucose = float(input("Glucose: "))
    blood_pressure = float(input("Blood Pressure: "))
    skin_thickness = float(input("Skin Thickness: "))
    insulin = float(input("Insulin: "))
    bmi = float(input("BMI: "))
    dpf = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))

    data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    data = diabetes_scaler.transform(data)

    prediction = diabetes_model.predict(data)

    if prediction[0] == 1:
        print("\nHIGH RISK OF DIABETES")
    else:
        print("\nLOW RISK OF DIABETES")

# ==========================================
# HEART DISEASE PREDICTION
# ==========================================

def heart_prediction():

    print("\nENTER HEART DISEASE DETAILS\n")

    age = float(input("Age: "))
    sex = float(input("Sex (0=Female,1=Male): "))
    cp = float(input("Chest Pain Type: "))
    trestbps = float(input("Resting Blood Pressure: "))
    chol = float(input("Cholesterol: "))
    fbs = float(input("Fasting Blood Sugar: "))
    restecg = float(input("Rest ECG: "))
    thalach = float(input("Max Heart Rate: "))
    exang = float(input("Exercise Angina: "))
    oldpeak = float(input("Oldpeak: "))
    slope = float(input("Slope: "))
    ca = float(input("CA: "))
    thal = float(input("Thal: "))

    data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    data = heart_scaler.transform(data)

    prediction = heart_model.predict(data)

    if prediction[0] == 1:
        print("\nHEART DISEASE DETECTED")
    else:
        print("\nNO HEART DISEASE DETECTED")

# ==========================================
# BREAST CANCER PREDICTION
# ==========================================

def breast_prediction():

    print("\nBREAST CANCER PREDICTION")

    print("\nEnter 30 Features")

    feature_names = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave_points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave_points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave_points_worst",
        "symmetry_worst",
        "fractal_dimension_worst"
    ]

    values = []

    for feature in feature_names:
        value = float(input(f"{feature}: "))
        values.append(value)

    data = np.array([values])

    data = breast_scaler.transform(data)

    prediction = breast_model.predict(data)

    if prediction[0] == 1:
        print("\nMALIGNANT CANCER DETECTED")
    else:
        print("\nBENIGN TUMOR")

# ==========================================
# MAIN MENU
# ==========================================

while True:

    print("\n" + "=" * 50)
    print("DISEASE PREDICTION SYSTEM")
    print("=" * 50)

    print("1. Diabetes Prediction")
    print("2. Heart Disease Prediction")
    print("3. Breast Cancer Prediction")
    print("4. Exit")

    choice = input("\nEnter Choice: ")

    if choice == "1":
        diabetes_prediction()

    elif choice == "2":
        heart_prediction()

    elif choice == "3":
        breast_prediction()

    elif choice == "4":
        print("\nThank You")
        break

    else:
        print("\nInvalid Choice")