import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@st.cache_resource
def load_and_train_model():
    # Load dataset
    url = https://github.com/Sameer455/heart_model/blob/main/heart.csv  # Update path
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
    st.success(f"Model trained successfully! Accuracy: {acc:.2f}")
    return model

def main():
    st.title("Heart Disease Prediction App")

    # Load model
    model = load_and_train_model()

    # Create input fields
    st.header("Patient Details")
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    sex = st.selectbox("Sex", options=["Female (0)", "Male (1)"], index=1)
    cp = st.selectbox("Chest Pain Type", options=["0", "1", "2", "3"])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No (0)", "Yes (1)"], index=0)
    restecg = st.selectbox("Resting ECG Results", options=["0", "1", "2"], index=0)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise-Induced Angina", options=["No (0)", "Yes (1)"], index=0)
    oldpeak = st.number_input("ST Depression Induced", min_value=0.0, max_value=6.0, value=1.0, format="%.1f")
    slope = st.selectbox("Slope of Peak Exercise", options=["0", "1", "2"], index=1)
    ca = st.selectbox("Number of Major Vessels", options=["0", "1", "2", "3", "4"], index=0)
    thal = st.selectbox("Thalassemia", options=["Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"], index=2)

    # Convert inputs to model-friendly format
    input_data = np.array([[age, int(sex[0]), int(cp), trestbps, chol, int(fbs[0]), int(restecg),
                            thalach, int(exang[0]), oldpeak, int(slope[0]), int(ca[0]), int(thal[0])]])

    # Predict
    if st.button("Predict"):
        prediction = model.predict(input_data)
        result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
        st.subheader(f"Prediction: {result}")

if __name__ == "__main__":
    main()
