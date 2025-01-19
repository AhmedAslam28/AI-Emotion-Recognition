# app.py
import streamlit as st
import requests
from PIL import Image
import io
import json

# Configure page layout
st.set_page_config(layout="wide")

# API endpoint
API_URL = "http://localhost:8000"

# Emotion GIFs paths
emotion_gifs = {
    "Angry": "gif\angry.gif",
    "Disgust": "gif\disgust.gif",
    "Fear": "gif\fear.gif",
    "Happy": "gifhappy.gif",
    "Sad": "gif\sadd.gif",
    "Surprise": "gif\surprise.gif",
    "Neutral": "gif\neutral.gif"
}

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
def main():
    st.title("Emotion Recognition with AI Assistant")

    # Create three main columns
    col1, col2, col3 = st.columns([1, 1, 1])

    # Left column: Image upload and emotion probabilities
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            image.thumbnail((150, 150))
            st.image(image, caption='Uploaded Image', width=150)
            
            # Send image to backend for processing
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{API_URL}/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                # Store prediction results in session state
                st.session_state.current_prediction = result
                
                # Display emotion probabilities
                st.write("Emotion Probabilities:")
                st.bar_chart(result["probabilities"], height=200)

    # Middle column: Predicted emotion and GIF
    with col2:
        if uploaded_file and hasattr(st.session_state, 'current_prediction'):
            result = st.session_state.current_prediction
            
            # Display predicted emotion and confidence
            st.write(f"**Predicted: {result['emotion']}** ({result['confidence']:.2f}%)")
            
            # Display emotion GIF
            if result['emotion'] in emotion_gifs:
                st.image(emotion_gifs[result['emotion']], width=500)
            
            # Book recommendations
            with st.expander("📚 Book Recommendations"):
                if "recommendations" in result:
                    for book in result["recommendations"]["Books"]:
                        st.write(f"• {book}")

    # Right column: AI Assistant and Chat
    with col3:
        if uploaded_file and hasattr(st.session_state, 'current_prediction'):
            result = st.session_state.current_prediction
            
            # AI Support
            with st.expander("🤖 AI Support", expanded=True):
                if "ai_support" in result:
                    st.write(result["ai_support"])
            
            # Chat interface
            with st.expander("💭 Chat History", expanded=True):
                # Chat input
                user_message = st.text_input("Ask about managing emotions:", key="chat_input")
                
                if user_message:
                    # Send message to backend
                    chat_response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "message": user_message,
                            "emotion": result["emotion"],
                            "confidence": result["confidence"]
                        }
                    )
                    
                    if chat_response.status_code == 200:
                        assistant_message = chat_response.json()["response"]
                        st.session_state.chat_history.append(("You", user_message))
                        st.session_state.chat_history.append(("Assistant", assistant_message))
                
                # Display chat history
                chat_container = st.container()
                with chat_container:
                    for role, message in st.session_state.chat_history[-5:]:
                        st.write(f"{'👤' if role == 'You' else '🤖'} **{role}:** {message}")

if __name__ == "__main__":
    main()