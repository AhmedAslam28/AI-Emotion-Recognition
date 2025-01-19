# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import google.generativeai as genai
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = tf.keras.models.load_model("emotion_reco.h5")

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyB5TrGgfFz51-YGmDqobDU76UOsZ-cPOL8"
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini chat
gemini_model = genai.GenerativeModel('gemini-pro')

# Constants
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

RECOMMENDATIONS = {
    "Happy": {"Books": ["The Happiness Project by Gretchen Rubin", "The Art of Happiness by Dalai Lama"]},
    "Sad": {"Books": ["Option B by Sheryl Sandberg", "When Breath Becomes Air by Paul Kalanithi"]},
    "Angry": {"Books": ["Anger: Wisdom for Cooling the Flames by Thich Nhat Hanh", "The Dance of Anger by Harriet Lerner"]},
    "Neutral": {"Books": ["Thinking, Fast and Slow by Daniel Kahneman", "Atomic Habits by James Clear"]},
    "Fear": {"Books": ["Daring Greatly by Brené Brown", "Feel the Fear and Do It Anyway by Susan Jeffers"]},
    "Surprise": {"Books": ["Surprise: Embrace the Unpredictable by Tania Luna", "The Power of Surprise by Michael Rousell"]},
    "Disgust": {"Books": ["Clean: The New Science of Skin by James Hamblin", "The Stuff of Thought by Steven Pinker"]}
}

class ChatMessage(BaseModel):
    message: str
    emotion: str
    confidence: float

def preprocess_image(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize and normalize
    image = cv2.resize(image, (48, 48)) / 255.0
    return np.reshape(image, (1, 48, 48, 1))

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    # Read and preprocess the image
    contents = await file.read()
    preprocessed_image = preprocess_image(contents)
    
    # Make prediction
    prediction = model.predict(preprocessed_image)
    emotion_percentages = prediction[0] * 100
    
    # Get predicted emotion and confidence
    emotion_index = np.argmax(prediction)
    emotion_label = EMOTION_LABELS[emotion_index]
    confidence = float(emotion_percentages[emotion_index])
    
    # Get recommendations
    recommendations = RECOMMENDATIONS.get(emotion_label, {"Books": []})
    
    # Get AI support
    chat = gemini_model.start_chat()
    support_prompt = f"I'm currently feeling {emotion_label} (confidence: {confidence:.2f}%). Provide brief, empathetic advice and coping strategies."
    support_response = chat.send_message(support_prompt)
    
    return {
        "emotion": emotion_label,
        "confidence": confidence,
        "probabilities": dict(zip(EMOTION_LABELS, emotion_percentages.tolist())),
        "recommendations": recommendations,
        "ai_support": support_response.text
    }

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    chat = gemini_model.start_chat()
    response = chat.send_message(chat_message.message)
    return {"response": response.text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)