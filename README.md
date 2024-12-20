import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_and_train_model():
    # Load dataset
    url = "C:\\Users\\shaik\\OneDrive\\Desktop\\project\\heart.csv"  # Update path
    df = pd.read_csv(url)

    # Define features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model trained successfully! Accuracy: {acc:.2f}")
    return model

def predict_heart_disease(model):
    print("Enter patient details for heart disease prediction:")
    age = int(input("Age: "))
    sex = int(input("Sex (0 = female, 1 = male): "))
    cp = int(input("Chest Pain Type (0-3): "))
    trestbps = int(input("Resting Blood Pressure: "))
    chol = int(input("Cholesterol: "))
    fbs = int(input("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False): "))
    restecg = int(input("Resting ECG Results (0-2): "))
    thalach = int(input("Max Heart Rate Achieved: "))
    exang = int(input("Exercise-Induced Angina (0 = No, 1 = Yes): "))
    oldpeak = float(input("ST Depression Induced: "))
    slope = int(input("Slope of Peak Exercise (0-2): "))
    ca = int(input("Number of Major Vessels (0-4): "))
    thal = int(input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect): "))

    # Prepare input features
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
    print(f"Prediction: {result}")

if __name__ == "__main__":
    # Load and train model
    rf_model = load_and_train_model()

    # Predict based on user input
    predict_heart_disease(rf_model)
