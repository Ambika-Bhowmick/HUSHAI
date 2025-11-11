# Import everything that is needed
import gradio as gr
import joblib
import numpy as np
import librosa

# Load models and encoders
model = joblib.load("material_recommender.pkl")
noise_encoder = joblib.load("noise_encoder.pkl")
material_encoder = joblib.load("material_encoder.pkl")

# Detect noise type from audio
def detect_noise(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
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

# Predict best soundproofing material
def predict_material(audio_path):
    if audio_path is None:
        return "No audio provided", "N/A"

    noise_label = detect_noise(audio_path)
    if noise_label == "Background noise":
        return noise_label, "N/A"

    encoded_noise = noise_encoder.transform([noise_label])
    input_features = np.array([[encoded_noise[0], 0]])  # placeholder feature
    predicted_material_encoded = model.predict(input_features)
    predicted_material = material_encoder.inverse_transform(predicted_material_encoded)[0]

    return noise_label, predicted_material

# Create Gradio interface
with gr.Blocks(title="H.U.S.H AI - Helping Us Silence Homes") as demo:
    gr.Markdown("## 🎧 H.U.S.H AI - Helping Us Silence Homes")
    gr.Markdown("Record or upload a sound to get the best soundproofing material recommendation.")
    
    audio_input = gr.Audio(type="filepath", label="🎙️ Record or Upload Sound")
    detect_btn = gr.Button("Analyze Sound")
    noise_output = gr.Textbox(label="Detected Noise Type")
    material_output = gr.Textbox(label="Recommended Material")
    
    detect_btn.click(predict_material, inputs=audio_input, outputs=[noise_output, material_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
