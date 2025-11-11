// Link for my teachable machine model
const MODEL_URL = "https://teachablemachine.withgoogle.com/models/fXQhRWczw/";

// Global Variables
let model, microphone, isListening = false;
const CONFIDENCE_THRESHOLD = 0.75; // Predictions can only be made with 60% confidence

// Funciton to load TM Model
async function loadModel() {
    const modelURL = MODEL_URL + "model.json"; // Model weight
    const metadataURL = MODEL_URL + "metadata.json"; // Info for classes and features
    model = await tmAudio.load(modelURL, metadataURL); // Load model with TM audio library
    console.log("TM model loaded");
}

// Function to start listining for microphone
async function startListening() {
    try {
      // Asking for mic permission
        await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log(" Microphone access granted");

      // Load the model if it's not loaded
        if (!model) await loadModel();

      //Get audio stream from mic
        microphone = new tmAudio.AudioStream();
        await microphone.open();
        console.log("✅ Microphone started");

        isListening = true; // If the isListening is true
        loop(); // Start the function called loop
    } catch (err) {
        console.error("🚫 Mic error:", err);
        alert("Please allow microphone access!");
    }
}

// Main part that checks what noise is being heard constantly
async function loop() {
    if (!isListening) return; // Stop when not listening

    const predictions = await model.predict(microphone);
    // Find the highest probability prediction
    let highest = predictions.reduce((a, b) => (a.probability > b.probability ? a : b));

    // Only send to backend if confident
    if (highest.probability >= CONFIDENCE_THRESHOLD) {
        const noiseLabel = highest.className;
        document.getElementById("noiseLabel").innerText = noiseLabel;

        // Send the noise label to the backend to get the recommended material
        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ noise: noiseLabel })
        })
        .then(res => res.json())
        .then(data => {
          // Update the material from backend response
            document.getElementById("materialLabel").innerText = data.material || data.error;
        });
    } else {
        // If confidence is low, show uncertainty
        document.getElementById("noiseLabel").innerText = "Background noise / uncertain";
        document.getElementById("materialLabel").innerText = "N/A";
    }
    // Call loop again
    requestAnimationFrame(loop);
}
// Connect the start listening button in HTML to startListening()
document.getElementById("startBtn").addEventListener("click", startListening);
