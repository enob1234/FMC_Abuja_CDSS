import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="FMC Abuja CDSS", layout="wide", page_icon="🩺")

# Hide Streamlit Deploy button and Main Menu (optional cleanliness)
hide_streamlit_style = """
<style>
    .stAppDeployButton {display:none;}
    .stDeployButton {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_path = 'pure_prior_xgboost.joblib'
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    return joblib.load(model_path)

model = load_model()

col1, col2 = st.columns([1, 6])
with col1:
    st.image("fmc_logo.png", use_container_width=True)
with col2:
    st.title("FMC Abuja Next-Visit SUH Risk Predictor")

st.markdown("""
Welcome to the Clinical Decision Support System prototype for FMC Abuja. 
This tool predicts whether a patient's blood pressure will remain **Sustained Uncontrolled (SUH)** at their **immediate next scheduled visit**.
""")

st.info("ℹ️ **Clinical Disclaimer:** This prediction and SHAP analysis is designed to guide and support the clinician, not to replace professional medical judgment. It serves as an assistive tool for clinical decision-making.")

# Input form
with st.sidebar:
    st.header("Patient Clinical Profile")
    age = st.number_input("Age (Years)", 18, 100, 55)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    bmi = st.number_input("BMI", 15.0, 50.0, 28.5)
    weight_kg = st.number_input("Weight (kg)", 40.0, 150.0, 80.0)
    height_cm = st.number_input("Height (cm)", 140.0, 200.0, 165.0)
    
    st.subheader("Blood Pressure History")
    sbp_previous = st.number_input("Previous Visit SBP", 90, 200, 145)
    dbp_previous = st.number_input("Previous Visit DBP", 60, 130, 92)
    sbp_mean_3visits = st.number_input("Mean SBP (Last 3 Visits)", 90, 200, 142)
    dbp_mean_3visits = st.number_input("Mean DBP (Last 3 Visits)", 60, 130, 89)
    bp_controlled_last_visit = st.selectbox("BP Controlled at Last Visit?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    st.subheader("Medications & Adherence")
    num_antihypertensive_drugs = st.number_input("Number of Anti-HTN Drugs", 0, 5, 2)
    months_on_current_regimen = st.number_input("Months on Current Regimen", 0, 120, 12)
    medication_class = st.selectbox("Primary Medication Class", options=[0, 3, 6], format_func=lambda x: {0: "CCB", 3: "ACEI/ARB", 6: "Diuretic"}.get(x, "Other"))
    days_since_last_visit = st.number_input("Days Since Last Visit", 1, 365, 30)
    total_visits_to_date = st.number_input("Total Visits to Date", 1, 100, 15)
    missed_appointment = st.selectbox("Recent Missed Appointment?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    appointment_adherence_rate = st.slider("Appointment Adherence Rate", 0.0, 1.0, 0.85)

    st.subheader("Comorbidities")
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    ckd = st.selectbox("CKD", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_failure = st.selectbox("Heart Failure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    stroke_history = st.selectbox("Stroke History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    obesity = 1 if bmi >= 30 else 0
    comorbidity_count = diabetes + ckd + heart_failure + stroke_history + obesity

    st.markdown("---")
    with st.expander("Admin Login"):
        admin_pwd = st.text_input("Enter Admin Password", type="password")
        
    if admin_pwd == "fmc_admin2026":
        st.subheader("Admin & Analytics")
        feedback_file = "clinical_feedback_results.csv"
        if os.path.exists(feedback_file):
            with open(feedback_file, "rb") as f:
                st.download_button("📥 Download Feedback Data", f, file_name="clinical_feedback_results.csv", mime="text/csv")
                
            with st.expander("📊 View Feedback Analytics"):
                try:
                    df_fb = pd.read_csv(feedback_file)
                    st.write(f"**Total Responses:** {len(df_fb)}")
                    
                    if len(df_fb) > 0 and "q1_accuracy" in df_fb.columns:
                        metrics = {
                            "Accuracy": df_fb["q1_accuracy"].mean(),
                            "Explainability": df_fb["q2_explainability"].mean(),
                            "Clinical Utility": df_fb["q3_utility"].mean(),
                            "Actionability": df_fb["q4_actionability"].mean(),
                            "Workflow": df_fb["q5_workflow"].mean(),
                            "Trust": df_fb["q6_trust"].mean()
                        }
                        df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Avg Score (1-5)"])
                        st.bar_chart(df_metrics.set_index("Metric"), height=250)
                    else:
                        st.info("No data yet for the new questions.")
                except Exception as e:
                    st.error("Analytics loading error.")

if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

if st.button("Predict Next-Visit Risk"):
    st.session_state.prediction_made = True

if st.session_state.prediction_made:
    if model is None:
        st.stop()
        
    med_0 = 1 if medication_class == 0 else 0
    med_3 = 1 if medication_class == 3 else 0
    med_6 = 1 if medication_class == 6 else 0
    
    # Feature dictionary
    features = {
        'sbp_previous': sbp_previous,
        'dbp_previous': dbp_previous,
        'sbp_mean_3visits': sbp_mean_3visits,
        'dbp_mean_3visits': dbp_mean_3visits,
        'bp_controlled_last_visit': bp_controlled_last_visit,
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'weight_kg': weight_kg,
        'height_cm': height_cm,
        'num_antihypertensive_drugs': num_antihypertensive_drugs,
        'months_on_current_regimen': months_on_current_regimen,
        'diabetes': diabetes,
        'ckd': ckd,
        'heart_failure': heart_failure,
        'stroke_history': stroke_history,
        'obesity': obesity,
        'comorbidity_count': comorbidity_count,
        'days_since_last_visit': days_since_last_visit,
        'total_visits_to_date': total_visits_to_date,
        'missed_appointment': missed_appointment,
        'appointment_adherence_rate': appointment_adherence_rate,
        'medication_class_0': med_0,
        'medication_class_3': med_3,
        'medication_class_6': med_6,
        'age_x_sbp_previous': age * sbp_previous,
        'prior_control_x_bmi': bp_controlled_last_visit * bmi,
        'prior_control_x_days': bp_controlled_last_visit * days_since_last_visit
    }
    
    # Needs exact column order as training
    expected_cols = [
        'sbp_previous', 'dbp_previous', 'sbp_mean_3visits', 'dbp_mean_3visits', 
        'bp_controlled_last_visit', 'age', 'sex', 
        'bmi', 'weight_kg', 'height_cm', 'num_antihypertensive_drugs', 
        'months_on_current_regimen', 'diabetes', 'ckd', 'heart_failure', 
        'stroke_history', 'obesity', 'comorbidity_count', 'days_since_last_visit', 
        'total_visits_to_date', 'missed_appointment', 'appointment_adherence_rate', 
        'medication_class_0', 'medication_class_3', 'medication_class_6',
        'age_x_sbp_previous', 'prior_control_x_bmi', 'prior_control_x_days'
    ]
    
    df_input = pd.DataFrame([features])[expected_cols]
    
    # Predict
    prob = model.predict_proba(df_input)[0][1]
    
    st.markdown("---")
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Prediction Result")
        risk_color = "red" if prob >= 0.5 else "green"
        st.markdown(f"<h1 style='color: {risk_color};'>{prob * 100:.1f}% Risk of Next-Visit SUH</h1>", unsafe_allow_html=True)
        if prob >= 0.5:
            st.warning("⚠️ This patient is at high risk of Sustained Uncontrolled Hypertension at their next visit. Consider therapeutic escalation or adherence counseling.")
        else:
            st.success("✅ This patient's blood pressure is predicted to be controlled at their next visit.")
            
    with cols[1]:
        st.subheader("Explainability (SHAP)")
        st.info("Generating personalized SHAP waterfall plot...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(df_input)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        
        st.markdown("""
        **How to read this plot:**
        - 🔴 **Red bars** represent patient factors pushing the risk of uncontrolled blood pressure **higher**.
        - 🔵 **Blue bars** represent protective factors pushing the risk **lower**.
        - **E[f(X)]** is the baseline average risk (log-odds) across the entire FMC Abuja patient population.
        - **f(x)** is the final predicted risk score (log-odds) for this specific patient before converting to a percentage.
        """)
        
        with st.expander("🔍 View All 28 Feature Contributions (Remove Blackbox)"):
            st.markdown("This table lists exactly how every single patient feature influenced the model's prediction, including the 19 features collapsed in the plot above.")
            shap_df = pd.DataFrame({
                "Feature": expected_cols,
                "Patient Value": df_input.iloc[0].values,
                "SHAP Contribution": shap_values[0].values
            })
            shap_df['Impact'] = shap_df['SHAP Contribution'].apply(lambda x: "⬆️ Increased Risk" if x > 0 else ("⬇️ Decreased Risk" if x < 0 else "Neutral"))
            shap_df['Absolute Impact'] = shap_df['SHAP Contribution'].abs()
            shap_df = shap_df.sort_values(by='Absolute Impact', ascending=False).drop(columns=['Absolute Impact'])
            st.dataframe(shap_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; margin-top: 10px; margin-bottom: 30px;">
            <a href="javascript:window.print()" style="display:inline-block; padding:12px 24px; background-color:#1b7a43; color:white; text-decoration:none; border-radius:6px; font-weight:bold; font-size:16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                🖨️ Print Clinical Report
            </a>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📝 Evaluation Feedback")
    st.markdown("Please provide your feedback on this prediction to help improve the CDSS prototype for FMC Abuja.")
    
    with st.form("feedback_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            q_accuracy = st.slider("1. I agree with the model's risk prediction.", 1, 5, 3)
            q_explainability = st.slider("2. The SHAP plot helped me understand the reasoning.", 1, 5, 3)
            q_utility = st.slider("3. This tool would help identify high-risk patients earlier.", 1, 5, 3)
        with col_f2:
            q_actionability = st.slider("4. The risk factors point to clear clinical interventions.", 1, 5, 3)
            q_workflow = st.slider("5. This tool is time-efficient for standard consultations.", 1, 5, 3)
            q_trust = st.slider("6. I would trust this system to support my decisions.", 1, 5, 3)
        
        q_comments = st.text_area("7. What change would you recommend to improve the system?")
        
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            feedback_data = pd.DataFrame([{
                "predicted_risk_prob": round(prob, 3),
                "q1_accuracy": q_accuracy,
                "q2_explainability": q_explainability,
                "q3_utility": q_utility,
                "q4_actionability": q_actionability,
                "q5_workflow": q_workflow,
                "q6_trust": q_trust,
                "comments": q_comments
            }])
            if not os.path.exists(feedback_file):
                feedback_data.to_csv(feedback_file, index=False)
            else:
                feedback_data.to_csv(feedback_file, mode='a', header=False, index=False)
            st.success("Thank you! Your feedback has been securely saved for the research study.")
