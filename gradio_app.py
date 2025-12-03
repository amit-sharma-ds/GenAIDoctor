import gradio as gr
import os
import pickle
import numpy as np
import onnxruntime as ort
from PIL import Image

# ------------------- Load Models -------------------
working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(f"{working_dir}/models/diabetes.pkl", "rb"))
heart_model = pickle.load(open(f"{working_dir}/models/heart.pkl", "rb"))

# Load ONNX Tumor model
tumor_session = ort.InferenceSession(f"{working_dir}/models/model.onnx")
input_name = tumor_session.get_inputs()[0].name
print("ONNX Input Name:", input_name)

# ------------------- AI Doctor Function -------------------
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts

system_prompt = """
You have to act as a professional doctor, i know you are not but this is for learning purpose.
What's in this image? Do you find anything wrong with it medically?
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in your response.
Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Keep your answer concise (max 2 sentences).
"""

def ai_doctor(audio_filepath, image_filepath):
    speech_to_text_output = transcribe_with_groq(
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
        audio_filepath=audio_filepath,
        stt_model="whisper-large-v3"
    )
    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=system_prompt + speech_to_text_output,
            encoded_image=encode_image(image_filepath),
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
    else:
        doctor_response = "No image provided for analysis."

    output_path = "final.mp3"
    text_to_speech_with_gtts(input_text=doctor_response, output_filepath=output_path)
    return speech_to_text_output, doctor_response, output_path

# ------------------- ML Prediction Functions -------------------
def diabetes_predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):
    user_input = [float(x) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]]
    pred = diabetes_model.predict([user_input])[0]
    return "Diabetic" if pred == 1 else "Not Diabetic"

def heart_predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    user_input = [float(x) for x in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    pred = heart_model.predict([user_input])[0]
    return "Heart Disease" if pred == 1 else "No Heart Disease"

# -------- Tumor Prediction --------
def tumor_predict(image):
    try:
        img = Image.open(image).convert("RGB").resize((224, 224))
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        pred = tumor_session.run(None, {input_name: img})[0]

        class_index = np.argmax(pred)
        classes = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
        return f"Prediction: {classes[class_index]}"

    except Exception as e:
        return f"ERROR: {str(e)}"

# ------------------- Gradio UI -------------------
with gr.Blocks(css="""
body {background-color: #e6f2ff; font-family: 'Arial', sans-serif;}
.gr-button {background: linear-gradient(to right, #4CAF50, #45a049) !important; color: white !important; font-weight: bold;}
""") as demo:

    with gr.Tabs():

        # AI Doctor Tab
        with gr.TabItem("AI Doctor"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(sources=["microphone"], type="filepath")
                    image_input = gr.Image(type="filepath")
                    doctor_button = gr.Button("Get Doctor Response")
                with gr.Column(scale=1):
                    st_text_output = gr.Textbox(label="Speech to Text", interactive=False)
                    doctor_text_output = gr.Textbox(label="Doctor's Response", interactive=False)
                    doctor_voice_output = gr.Audio(label="Doctor Voice", interactive=False)

            doctor_button.click(ai_doctor, [audio_input, image_input],
                                [st_text_output, doctor_text_output, doctor_voice_output])

        # Diabetes Tab
        with gr.TabItem("Diabetes Prediction"):
            with gr.Row():
                with gr.Column():
                    Pregnancies = gr.Textbox(label="Pregnancies")
                    Glucose = gr.Textbox(label="Glucose")
                    BloodPressure = gr.Textbox(label="Blood Pressure")
                    SkinThickness = gr.Textbox(label="Skin Thickness")
                    Insulin = gr.Textbox(label="Insulin")
                    BMI = gr.Textbox(label="BMI")
                    DPF = gr.Textbox(label="DPF")
                    Age = gr.Textbox(label="Age")
                    diabetes_button = gr.Button("Check Diabetes")
                with gr.Column():
                    diabetes_output = gr.Textbox(label="Result", interactive=False)
            diabetes_button.click(diabetes_predict,
                                  [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age],
                                  diabetes_output)

        # Heart Tab
        with gr.TabItem("Heart Prediction"):
            with gr.Row():
                with gr.Column():
                    age = gr.Textbox(label="Age")
                    sex = gr.Textbox(label="Sex")
                    cp = gr.Textbox(label="Chest Pain Types")
                    trestbps = gr.Textbox(label="Resting BP")
                    chol = gr.Textbox(label="Cholesterol")
                    fbs = gr.Textbox(label="Fasting Blood Sugar")
                    restecg = gr.Textbox(label="Resting ECG")
                    thalach = gr.Textbox(label="Max Heart Rate")
                    exang = gr.Textbox(label="Exercise Induced Angina")
                    oldpeak = gr.Textbox(label="ST Depression")
                    slope = gr.Textbox(label="Slope")
                    ca = gr.Textbox(label="Major Vessels")
                    thal = gr.Textbox(label="Thalassemia")
                    heart_button = gr.Button("Check Heart Disease")
                with gr.Column():
                    heart_output = gr.Textbox(label="Result", interactive=False)

            heart_button.click(heart_predict,
                               [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal],
                               heart_output)

        # Tumor Tab
        with gr.TabItem("Tumor Prediction"):
            with gr.Row():
                with gr.Column():
                    tumor_image = gr.Image(type="filepath")
                    tumor_button = gr.Button("Check Tumor")
                with gr.Column():
                    tumor_output = gr.Textbox(label="Result", interactive=False)
            tumor_button.click(tumor_predict, tumor_image, tumor_output)

if __name__ == "__main__":
    demo.launch()
