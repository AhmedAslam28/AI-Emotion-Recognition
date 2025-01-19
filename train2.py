import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define paths to the training and testing folders
train_dir = "C:\\Users\\Aslam\\Generative AN\\FER\\train"  # Path to train folder
test_dir = "C:\\Users\\Aslam\\Generative AN\\FER\\test"    # Path to test folder

# Define image dimensions and other parameters
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
NUM_CLASSES = 7  # Assuming 7 emotion classes in FER-2013 dataset

# Use ImageDataGenerator to load and preprocess images from directory
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="sparse"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="sparse"
)

# Define the CNN + LSTM Model
def build_emotion_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES):
    model = models.Sequential()
    # CNN Layers
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and reshape for LSTM layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))  # Reduce dimensions before LSTM
    model.add(layers.Reshape((1, 128)))  # Reshape to (1, 128) for LSTM compatibility

    # LSTM Layer
    model.add(layers.LSTM(64, activation='relu', return_sequences=False))
    model.add(layers.Dropout(0.5))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and compile the model
model = build_emotion_model()
model.summary()

# Train the Model with increased epochs
EPOCHS = 25
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)

# Save the trained model
model.save("emotion_reco.h5")
print("Model trained and saved as emotion_recognition_model.h5")
