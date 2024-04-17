import streamlit as st
from PIL import Image
import zipfile
import os
import shutil
import torch
from torchvision import transforms

# Function to process the uploaded image
def process_image(image):
    # Load the trained model
    with zipfile.ZipFile('trained_model.zip', 'r') as zip_ref:
        zip_ref.extractall('trained_model')
    
    # Load the model
    model = torch.load('trained_model/model.pth')
    model.eval()

    # Load and preprocess the uploaded image
    img = Image.open(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    img = img.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(img)

    # Convert output back to image
    processed_image = output.squeeze(0).permute(1, 2, 0).numpy() * 255
    processed_image = Image.fromarray(processed_image.astype('uint8'))

    # Clean up extracted model files
    shutil.rmtree('trained_model')

    return processed_image

# Streamlit frontend
def main():
    st.title("Image Processing with Trained Model")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Run"):
            processed_image = process_image(uploaded_file)
            st.image(processed_image, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()
