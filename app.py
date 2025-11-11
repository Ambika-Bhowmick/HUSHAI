import gradio as gr
import joblib
import numpy as np
import librosa

# --- Load models and encoders ---
model = joblib.load("material_recommender.pkl")
noise_encoder = joblib.load("noise_encoder.pkl")
material_encoder = joblib.load("material_encoder.pkl")

# --- Map from audio classification to label ---
# These are your Teachable Machine class names
tm_labels = ["Background noise", "Construction", "Traffic", "Barking", "Neighbours"]

# --- Fake TM function (replace with your Teachable Machine later) ---
# For now it simulates noise detection from audio
def detect_noise(audio):
    # Load audio features (simulate TM classification)
    y, sr = librosa.load(audio, sr=None)
    rms = np.mean(librosa.feature.rms(y=y))
    if rms < 0.005:
        return "Background noise"
    elif rms < 0.02:
        return "Neighbours"
    elif rms < 0.04:
        return "Barking"
    elif rms < 0.06:
        return "Traffic"
    else:
        return "Construction"

# --- Prediction function ---
def predict_material(audio):
    if audio is None:
        return "No audio provided", "N/A"

    # Step 1: detect noise type from audio
    noise_label = detect_noise(audio)

    # Step 2: if background noise, return N/A
    if noise_label == "Background noise":
        return noise_label, "N/A"

    # Step 3: encode + predict material
    encoded_noise = noise_encoder.transform([noise_label])
    input_features = [[encoded_noise[0], 0]]  # dummy 2nd feature
    predicted_material_encoded = model.predict(input_features)
    predicted_material = material_encoder.inverse_transform(predicted_material_encoded)[0]

    return noise_label, predicted_material

# --- Build Gradio Interface ---
interface = gr.Interface(
    fn=predict_material,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Record or Upload Sound"),
    outputs=[
        gr.Textbox(label="Detected Noise"),
        gr.Textbox(label="Recommended Material")
    ],
    title="H.U.S.H AI - Helping Us Silence Homes",
    description="Record a sound and get the best soundproofing material recommendation."
)

# --- Launch for Hugging Face Spaces ---
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
