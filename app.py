import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Phone Addiction Predictor",
    page_icon="📱",
    layout="centered"
)

# --- MODEL LOADING LOGIC ---
@st.cache_resource
def load_trained_model():
    """
    Loads the trained model pipeline. 
    Checks for a local fallback file if MLflow artifacts aren't natively stored locally.
    """
    # Replace 'GradientBoosting_final_model.pkl' with whatever your best model file is named
    model_filename = "GradientBoosting_final_model.pkl" 
    
    if os.path.exists(model_filename):
        return joblib.load(model_filename)
    else:
        st.error(f"⚠️ Model file '{model_filename}' not found. Please run your training script first to generate it.")
        return None

model_pipeline = load_trained_model()

# --- APP INTERFACE ---
st.title("📱 Phone Addiction Level Predictor")
st.markdown("""
This app predicts an individual's **Phone Addiction Level** using psychological indicators, 
demographics, and phone usage configurations. Fill out the details below to see the evaluation.
""")

st.write("---")

# --- USER INPUT FORM ---
if model_pipeline is not None:
    # We use a form to avoid app refreshes on every single user interaction
    with st.form("prediction_form"):
        st.subheader("👥 Demographic Information")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=5, max_value=100, value=20, step=1)
        with col2:
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])

        st.write("---")
        st.subheader("📊 Daily Screen Time & Habits")
        
        col3, col4 = st.columns(2)
        with col3:
            daily_usage = st.number_input("Daily Usage (Hours)", min_value=0.0, max_value=24.0, value=4.5, step=0.1)
            weekend_usage = st.number_input("Weekend Usage (Hours)", min_value=0.0, max_value=24.0, value=6.0, step=0.1)
            screen_time_bed = st.number_input("Screen Time Before Bed (Hours)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
            phone_checks = st.number_input("Phone Checks Per Day", min_value=0, max_value=1000, value=80, step=1)
        
        with col4:
            apps_used = st.number_input("Apps Used Daily", min_value=0, max_value=100, value=15, step=1)
            social_media_time = st.number_input("Time on Social Media (Hours)", min_value=0.0, max_value=24.0, value=2.0, step=0.1)
            gaming_time = st.number_input("Time on Gaming (Hours)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
            education_time = st.number_input("Time on Education (Hours)", min_value=0.0, max_value=24.0, value=1.5, step=0.1)

        purpose = st.selectbox("Primary Phone Usage Purpose", options=['Browsing', 'Education', 'Social Media', 'Gaming', 'Other'])

        st.write("---")
        st.subheader("🧠 Well-being & Lifestyle Metrics")
        
        col5, col6 = st.columns(2)
        with col5:
            sleep_hours = st.number_input("Sleep Duration (Hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
            exercise_hours = st.number_input("Exercise Duration (Hours)", min_value=0.0, max_value=12.0, value=0.5, step=0.1)
            intellectual_perf = st.slider("Intellectual Performance Rating", min_value=0, max_value=100, value=75)
            
        with col6:
            social_interactions = st.slider("Social Interactions Level", min_value=0, max_value=10, value=5)
            family_communication = st.slider("Family Communication Level", min_value=0, max_value=10, value=6)
            self_esteem = st.slider("Self Esteem Level", min_value=1, max_value=10, value=7)

        col7, col8 = st.columns(2)
        with col7:
            anxiety = st.slider("Anxiety Level", min_value=0, max_value=10, value=3)
        with col8:
            depression = st.slider("Depression Level", min_value=0, max_value=10, value=3)

        # Submit button
        submit_btn = st.form_submit_with_button_style(
            label="🎯 Calculate Addiction Score", 
            type="primary"
        )

    # --- INFERENCE LOGIC ---
    if submit_btn:
        # Create a dictionary matching exactly the training dataset structure
        input_data = {
            'Age': age,
            'Gender': gender,
            'Daily_Usage_Hours': daily_usage,
            'Sleep_Hours': sleep_hours,
            'Interllectual_Performance': intellectual_perf,
            'Social_Interactions': social_interactions,
            'Exercise_Hours': exercise_hours,
            'Anxiety_Level': anxiety,
            'Depression_Level': depression,
            'Self_Esteem': self_esteem,
            'Screen_Time_Before_Bed': screen_time_bed,
            'Phone_Checks_Per_Day': phone_checks,
            'Apps_Used_Daily': apps_used,
            'Time_on_Social_Media': social_media_time,
            'Time_on_Gaming': gaming_time,
            'Time_on_Education': education_time,
            'Phone_Usage_Purpose': purpose,
            'Family_Communication': family_communication,
            'Weekend_Usage_Hours': weekend_usage
        }
        
        # Turn into a single row Dataframe
        input_df = pd.DataFrame([input_data])
        
        # NOTE: pd.get_dummies happens natively inside your training script. 
        # But because we used pipeline tracking, Scikit-Learn pipelines require all columns to look EXACTLY 
        # identical to how X_train looked post-dummies. Let's replicate those expected column structures.
        
        # List of categorical dummy columns created during training
        expected_categorical_dummies = [
            'Gender_Male', 'Gender_Other',
            'Phone_Usage_Purpose_Education', 'Phone_Usage_Purpose_Gaming', 
            'Phone_Usage_Purpose_Other', 'Phone_Usage_Purpose_Social Media'
        ]
        
        # Process the frontend single row inputs into structural One-Hot column encodings
        input_df['Gender_Male'] = 1 if gender == "Male" else 0
        input_df['Gender_Other'] = 1 if gender == "Other" else 0
        
        input_df['Phone_Usage_Purpose_Education'] = 1 if purpose == "Education" else 0
        input_df['Phone_Usage_Purpose_Gaming'] = 1 if purpose == "Gaming" else 0
        input_df['Phone_Usage_Purpose_Social Media'] = 1 if purpose == "Social Media" else 0
        input_df['Phone_Usage_Purpose_Other'] = 1 if purpose == "Other" else 0
        
        # Safely drop old string tracking columns
        input_df = input_df.drop(columns=['Gender', 'Phone_Usage_Purpose'])
        
        try:
            # Predict using the loaded final scikit-learn pipeline
            prediction = model_pipeline.predict(input_df)[0]
            
            # Show customized diagnostic results
            st.write("---")
            st.subheader("📊 Prediction Results")
            
            # Format display with a color metric alert block
            st.metric(label="Predicted Addiction Level Score", value=f"{prediction:.2f} / 10.0")
            
            if prediction >= 7.5:
                st.error("🚨 **High Addiction Risk:** The metric suggests a critical dependence pattern. Interventions or screen limits are strongly recommended.")
            elif 4.0 <= prediction < 7.5:
                st.warning("⚠️ **Moderate Addiction Risk:** Normal daily habits starting to encroach upon regular well-being parameters.")
            else:
                st.success("✅ **Healthy Usage Pattern:** Screen configurations appear completely balanced with real-world health parameters.")
                
        except Exception as e:
            st.error(f"Error compiling prediction structure: {e}")