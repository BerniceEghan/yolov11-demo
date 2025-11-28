"""
Streamlit Application for YOLOv11m Inference

This script loads a YOLOv11 model, accepts image uploads,
performs prediction, and displays annotated outputs.

Author:
    Bernice Eghan

Style:
    Google Python Style Guide
"""

import os
import tempfile
import requests
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


def download_model_from_gdrive(url, output_path):
    """Download YOLO model file from Google Drive.

    Args:
        url (str): Public Google Drive download link.
        output_path (str): Path where the model will be saved.

    Returns:
        str: Path to the saved model file.
    """
    st.info("Downloading YOLO model. Please wait...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    st.success("Model downloaded successfully.")
    return output_path


def load_model(model_path):
    """Load YOLO model from disk.

    Args:
        model_path (str): Path to model weights.

    Returns:
        YOLO: Loaded YOLO model object.
    """
    return YOLO(model_path)


def run_inference(model, image):
    """Run inference on uploaded image.

    Args:
        model (YOLO): Loaded YOLO model.
        image (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: Annotated prediction image.
    """
    results = model.predict(image, save=False)
    return results[0].plot()


def main():
    """Main Streamlit application function."""
    st.title("YOLOv11m Object Detection Demo")
    st.write("Upload an image to run detection using your trained YOLOv11m model.")

    # Google Drive model direct link
    drive_url = (
        "https://drive.google.com/uc?export=download&id=17w0ZL4swI_"
        "zLoqwgToa2j6CSei_EH7Tl"
    )

    model_path = "best.pt"

    # Download model only if not present
    if not Path(model_path).exists():
        download_model_from_gdrive(drive_url, model_path)

    # Load YOLO model
    model = load_model(model_path)

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Input Image")
        st.image(input_image, use_column_width=True)

        # Run inference
        with st.spinner("Running detection..."):
            output_image = run_inference(model, input_image)

        st.subheader("Detection Results")
        st.image(output_image, use_column_width=True)


if __name__ == "__main__":
    main()
