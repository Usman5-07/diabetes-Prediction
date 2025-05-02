import numpy as np 
import pickle 
import streamlit as st

model = pickle.load(open('D:/_Portfolio/ML Coding Projects/Diebeties Project/Diebeties_Trained_Model.sav', 'rb'))



def prediction(inputData):
    asNP = np.asarray(inputData)
    reshaped = asNP.reshape(1, -1)
    prediction = model.predict(reshaped)
    if prediction == 0:
        return "The person is NON-DIEBETIC"
    else:
        return "The person is DIEBETIC"

def main():
    st.title("Diabetes Prediction System")

    preg = st.text_input("No. of Pregnencies")
    glucose = st.text_input("Glucose Level")
    bp = st.text_input("Blood Pressure")
    skinThicknees = st.text_input("Skin Thickness")
    insulin = st.text_input("Insulin Level In body")
    bmi = st.text_input("BMI")
    diabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    age = st.text_input("Age")

    diagnosis = ''

    if st.button("Result"):
        diagnosis = prediction([preg, glucose, bp, skinThicknees, insulin, bmi, diabetesPedigreeFunction, age])

    st.success(diagnosis)

if __name__ == "__main__":
    main()
    