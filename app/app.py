import gradio as gr
import pandas as pd
import pickle

# Load the trained model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

label_mapping = {0: "Not Risk", 1: "Risky"}

age_category_mapping = {
    '18-24': 0,
    '25-29': 1,
    '30-34': 2,
    '35-39': 3,
    '40-44': 4,
    '45-49': 5,
    '50-54': 6,
    '55-59': 7,
    '60-64': 8,
    '65-69': 9,
    '70-74': 10,
    '75-79': 11,
    '80 or older': 12
}
race_mapping = {
    'American Indian/Alaskan Native': 0,
    'Asian': 1,
    'Black': 2,
    'Hispanic': 3,
    'Other': 4,
    'White': 5
}

diabetic_mapping = {
    'No': 0,
    'No, borderline diabetes': 0,
    'Yes': 1,
    'Yes (during pregnancy)': 1
}

gen_health_mapping = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very good': 3,
    'Excellent': 4
}

# Define a function to preprocess input data and make predictions
def predict_action(
    BMI,
    Smoking,
    AlcoholDrinking,
    Stroke,
    PhysicalHealth,
    MentalHealth,
    DiffWalking,
    Sex,
    AgeCategory,
    Race,
    Diabetic,
    PhysicalActivity,
    GenHealth,
    SleepTime,
    Asthma,
    KidneyDisease,
    SkinCancer,
):
    # Prepare input data as a DataFrame
    data = pd.DataFrame(
        {
            "BMI": [float(BMI)],
            "Smoking": [Smoking == 'Yes'],
            "AlcoholDrinking": [AlcoholDrinking == 'Yes'],
            "Stroke": [Stroke == 'Yes'],
            "PhysicalHealth": [int(PhysicalHealth)],
            "MentalHealth": [int(MentalHealth)],
            "DiffWalking": [DiffWalking == 'Yes'],
            "Sex": [0 if Sex == 'Male' else 1],  # Encoding Male as 0, Female as 1
            "AgeCategory": [age_category_mapping[AgeCategory]],
            "Race": [race_mapping[Race]],
            "Diabetic": [diabetic_mapping[Diabetic]],
            "PhysicalActivity": [PhysicalActivity == 'Yes'],
            "GenHealth": [gen_health_mapping[GenHealth]],
            "SleepTime": [float(SleepTime)],
            "Asthma": [Asthma == 'Yes'],
            "KidneyDisease": [KidneyDisease == 'Yes'],
            "SkinCancer": [SkinCancer == 'Yes'],
        }
    )

    # Make predictions using the loaded model
    prediction = model.predict(data)[0]
    decoded_prediction = label_mapping[prediction]
    return decoded_prediction

# Create Gradio interface with mandatory fields
iface = gr.Interface(
    fn=predict_action,
    inputs=[
        gr.Number(label="BMI"),
        gr.Radio(label="Smoking", choices=["Yes", "No"]),
        gr.Radio(label="Alcohol Drinking", choices=["Yes", "No"]),
        gr.Radio(label="Stroke", choices=["Yes", "No"]),
        gr.Number(label="Physical Health (days unhealthy in last 30 days)"),
        gr.Number(label="Mental Health (days unhealthy in last 30 days)"),
        gr.Radio(label="Difficulty Walking", choices=["Yes", "No"]),
        gr.Radio(label="Sex", choices=["Male", "Female"]),
        gr.Dropdown(label="Age Category", choices=list(age_category_mapping.keys())),
        gr.Dropdown(label="Race", choices=list(race_mapping.keys())),
        gr.Radio(label="Diabetic", choices=["Yes", "No", "No, but pre-diabetic"]),
        gr.Radio(label="Physical Activity", choices=["Yes", "No"]),
        gr.Dropdown(label="General Health", choices=["Excellent", "Very good", "Good", "Fair", "Poor"]),
        gr.Number(label="Sleep Time (hours)"),
        gr.Radio(label="Asthma", choices=["Yes", "No"]),
        gr.Radio(label="Kidney Disease", choices=["Yes", "No"]),
        gr.Radio(label="Skin Cancer", choices=["Yes", "No"])
    ],
    outputs="text",
    title="Heart Disease Prediction",
    description="Predict the possibility of heart disease based on provided data."
)

# Launch the interface
iface.launch()
