import os
import cv2
from transformers import pipeline # type: ignore
from PIL import Image

# Set up the Hugging Face pipeline for Indian Sign Language Classification
pipe = pipeline("image-classification", model="Hemg/Indian-sign-language-classification")

# Define function to predict the character in an image using the Hugging Face pipeline
def predict_character(image_path):
    # Load the image directly and convert it to RGB (as the model expects RGB format)
    pil_image = Image.open(image_path).convert("RGB")

    # Predict the character using the Hugging Face pipeline
    predictions = pipe(pil_image)

    # Get the label with the highest score
    predicted_label = predictions[0]["label"]
    
    return predicted_label

# Example usage
image_path = "C:/Users/ASUS/Desktop/PROJECT1/archive/data/1/0.jpg"  # Replace with the actual path to an image
predicted_character = predict_character(image_path)
print(f"The predicted character is: {predicted_character}")


