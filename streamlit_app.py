#[python -m streamlit run streamlit_app.py] run this for streamlit link
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('final_diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit page config
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🩺 Diabetes Risk Prediction App")
st.markdown("Enter your health information to assess your risk of diabetes.")

# Input fields (8 medically relevant)
glucose = st.number_input("Glucose Level", min_value=0, value=100)
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, value=25.0)
age = st.number_input("Age", min_value=1, value=30)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, step=0.01)
pregnancies = st.number_input("Number of Pregnancies", min_value=0, value=1)
insulin = st.number_input("Insulin Level", min_value=0.0, value=85.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
time = st.number_input("Time (duration metric)", min_value=0, value=100)

# Predict button
if st.button("Predict"):
    # Input in same order as training
    input_data = np.array([[glucose, bmi, age, dpf,
                            pregnancies, insulin, skin_thickness, time]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Display result
    if prediction == 1:
        st.error("⚠️ You may be at **risk of diabetes**.")
        st.markdown("💡 _Please consult a healthcare professional for confirmation._")

        # Lifestyle suggestions
        st.markdown("---")
        st.markdown("### 🩺 Lifestyle Suggestions for Managing Diabetes")

        st.markdown("#### 🍽️ Healthy Eating")
        st.markdown("- Focus on high-fiber vegetables, whole grains, and lean proteins.")
        st.markdown("- Avoid sugary foods, white rice/bread, and fried items.")
        st.markdown("- Eat smaller meals every 3–4 hours to control blood sugar.")

        st.markdown("#### 🏃‍♀️ Stay Active")
        st.markdown("- Aim for **30 minutes of physical activity daily**.")
        st.markdown("- Activities: brisk walking, cycling, yoga, swimming.")

        st.markdown("#### ⚖️ Weight Management")
        st.markdown("- Losing just 5–10% of weight can improve blood sugar control.")

        st.markdown("#### 💧 Stay Hydrated & Reduce Stress")
        st.markdown("- Drink water regularly. Avoid sugary drinks.")
        st.markdown("- Practice deep breathing, meditation, or light yoga.")

        st.markdown("#### 💊 Medication & Monitoring")
        st.markdown("- Follow your doctor's advice and take medications on time.")
        st.markdown("- Regularly monitor your blood glucose levels.")

        st.markdown("#### 👟 Foot & Skin Care")
        st.markdown("- Keep feet clean and dry. Moisturize daily.")
        st.markdown("- Check for cuts or sores regularly.")

    else:
        st.success("✅ You are likely **not at risk** of diabetes.")

    st.markdown("---")
    st.markdown("🧠 _This is a predictive tool, not a clinical diagnosis._")
