# AI-Emotion-Recognition
Emotion Recognition with AI Assistant
# Emotion Recognition System

This project integrates an end-to-end emotion recognition system using a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks, a FastAPI backend for emotion detection, and a Streamlit frontend for user interaction. It processes grayscale images to classify emotions into seven categories, provides book recommendations, AI-powered advice, and an interactive chat interface.

---

## Features
### Model
- **Image Preprocessing**: Uses TensorFlow's `ImageDataGenerator` to preprocess and augment image data.
- **CNN + LSTM Architecture**:
  - 3 convolutional layers for feature extraction.
  - LSTM layer with 64 units for temporal sequence learning.
  - Fully connected layer with `softmax` activation for classification.
- **Model Output**: The trained model is saved as `emotion_reco.h5`.

### Backend (FastAPI)
- **Emotion Prediction**: Predicts emotions from images using the trained model.
- **Book Recommendations**: Suggests books based on the predicted emotion.
- **AI Support**: Provides empathetic advice using the Google Generative AI (Gemini API).
- **Chat Interface**: Supports user queries with AI-generated responses.

### Frontend (Streamlit)
- **Image Upload**: Allows users to upload images for emotion detection.
- **Emotion Display**: Shows predicted emotion, confidence, and corresponding GIF.
- **Probabilities Chart**: Displays a bar chart of emotion prediction probabilities.
- **Recommendations and Chat**: Offers personalized book suggestions and an AI chat interface.

---

## Prerequisites
- **Programming Language**: Python 3.11
- **Dependencies**:
  - TensorFlow >= 2.x
  - FastAPI
  - Uvicorn
  - OpenCV
  - Pillow
  - Streamlit
  - Google Generative AI SDK

---

##Project Implementation
### 1. Start the Backend Server
Run the FastAPI backend server:

```bash
uvicorn back:app --host 0.0.0.0 --port 8000



