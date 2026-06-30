import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =====================================================
# COMMON TRAINING FUNCTION
# =====================================================

def train_and_select_best_model(X, y, disease_name):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression":
            LogisticRegression(max_iter=2000),

        "SVM":
            SVC(probability=True),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

        "XGBoost":
            XGBClassifier(
                eval_metric="logloss",
                random_state=42
            )
    }

    best_model = None
    best_accuracy = 0

    print("\n" + "=" * 60)
    print(f"{disease_name} MODEL RESULTS")
    print("=" * 60)

    for name, model in models.items():

        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(f"{name:<25} Accuracy : {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print("\nBest Accuracy :", round(best_accuracy, 4))

    best_predictions = best_model.predict(
        X_test_scaled
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            best_predictions
        )
    )

    return best_model, scaler

# =====================================================
# DIABETES DATASET
# =====================================================

print("\nLoading Diabetes Dataset...")

diabetes_columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

diabetes_df = pd.read_csv(
    "datasets/diabetes.csv",
    names=diabetes_columns
)

X_diabetes = diabetes_df.drop(
    "Outcome",
    axis=1
)

y_diabetes = diabetes_df["Outcome"]

diabetes_model, diabetes_scaler = train_and_select_best_model(
    X_diabetes,
    y_diabetes,
    "DIABETES"
)

joblib.dump(
    diabetes_model,
    "models/diabetes_model.pkl"
)

joblib.dump(
    diabetes_scaler,
    "scalers/diabetes_scaler.pkl"
)

print("\nDiabetes Model Saved")

# =====================================================
# HEART DISEASE DATASET
# =====================================================

print("\nLoading Heart Disease Dataset...")

heart_df = pd.read_csv(
    "datasets/heart.csv"
)

X_heart = heart_df.drop(
    "target",
    axis=1
)

y_heart = heart_df["target"]

heart_model, heart_scaler = train_and_select_best_model(
    X_heart,
    y_heart,
    "HEART DISEASE"
)

joblib.dump(
    heart_model,
    "models/heart_model.pkl"
)

joblib.dump(
    heart_scaler,
    "scalers/heart_scaler.pkl"
)

print("\nHeart Disease Model Saved")

# =====================================================
# BREAST CANCER DATASET
# =====================================================

print("\nLoading Breast Cancer Dataset...")

breast_df = pd.read_csv(
    "datasets/breast_cancer.csv"
)

# Remove unnecessary columns

if "id" in breast_df.columns:
    breast_df.drop(
        columns=["id"],
        inplace=True
    )

for col in breast_df.columns:
    if "Unnamed" in col:
        breast_df.drop(
            columns=[col],
            inplace=True
        )

encoder = LabelEncoder()

breast_df["diagnosis"] = encoder.fit_transform(
    breast_df["diagnosis"]
)

X_breast = breast_df.drop(
    "diagnosis",
    axis=1
)

y_breast = breast_df["diagnosis"]

breast_model, breast_scaler = train_and_select_best_model(
    X_breast,
    y_breast,
    "BREAST CANCER"
)

joblib.dump(
    breast_model,
    "models/breast_model.pkl"
)

joblib.dump(
    breast_scaler,
    "scalers/breast_scaler.pkl"
)

print("\nBreast Cancer Model Saved")

# =====================================================
# FINISH
# =====================================================

print("\n" + "=" * 60)
print("ALL MODELS TRAINED SUCCESSFULLY")
print("MODELS SAVED IN models FOLDER")
print("SCALERS SAVED IN scalers FOLDER")
print("=" * 60)