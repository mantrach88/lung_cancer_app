import streamlit as st
import numpy as np
from model import sigmoid, w_opt, scaler

st.title("Lung Cancer Risk Survey")

st.write("Please fill in the following information:")

# Basic info
age = st.slider("Age", 10, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
if gender == "Female":
    gender = 0
else: gender = 1

# Features (1 to 10)
air_pollution = st.select_slider(
    "Air Pollution Exposure (1 = Low, 10 = High)",
    options=list(range(1, 11)),
    value=5,
    help="Low = 1-3, Medium = 4-7, High = 8-10"
)
alcohol_use = st.slider("Alcohol Use", 1, 10, 5)
dust_allergy = st.slider("Dust Allergy", 1, 10, 5)
occupational_hazards = st.slider("Occupational Hazards", 1, 10, 5)
genetic_risk = st.slider("Genetic Risk", 1, 10, 5)
chronic_lung_disease = st.slider("Chronic Lung Disease", 1, 10, 5)
balanced_diet = st.slider("Balanced Diet", 1, 10, 5)
obesity = st.slider("Obesity", 1, 10, 5)
smoking = st.slider("Smoking", 1, 10, 5)
passive_smoker = st.slider("Passive Smoker Exposure", 1, 10, 5)
chest_pain = st.slider("Chest Pain", 1, 10, 5)
coughing_blood = st.slider("Coughing Blood", 1, 10, 5)
fatigue = st.slider("Fatigue", 1, 10, 5)
weight_loss = st.slider("Weight Loss", 1, 10, 5)
shortness_of_breath = st.slider("Shortness of Breath", 1, 10, 5)
wheezing = st.slider("Wheezing", 1, 10, 5)
swallowing_difficulty = st.slider("Swallowing Difficulty", 1, 10, 5)
clubbing = st.slider("Clubbing of Finger Nails", 1, 10, 5)
frequent_cold = st.slider("Frequent Cold", 1, 10, 5)
dry_cough = st.slider("Dry Cough", 1, 10, 5)
snoring = st.slider("Snoring", 1, 10, 5)
user_input = [age, gender, air_pollution, alcohol_use, dust_allergy, occupational_hazards, genetic_risk, 
              chronic_lung_disease, balanced_diet, obesity, smoking, passive_smoker, chest_pain, coughing_blood,
              fatigue, weight_loss, shortness_of_breath, wheezing, swallowing_difficulty, clubbing, frequent_cold,
              dry_cough, snoring]


if st.button("Submit"):
    st.success("✅ Submitted!")
    user_input_np = np.array(user_input).reshape(1, -1)
    user_scaled = scaler.transform(user_input_np)
    user_with_bias = np.hstack([user_scaled, [[1]]])
    
    prob = sigmoid(user_with_bias @ w_opt)[0]
    prediction = "Lung Cancer" if prob > 0.5 else "No Lung Cancer"
    
    st.markdown("---")
    st.subheader("Prediction Result")
    st.write(f"**Probability of Lung Cancer:** {prob:.2%}")
    st.write(f"**Prediction:** {prediction}")
