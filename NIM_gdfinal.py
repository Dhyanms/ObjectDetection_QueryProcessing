import time
import uuid
import zipfile
import requests
import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# NVIDIA API endpoints and authorization
nvai_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nv-grounding-dino"
nvai_polling_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
header_auth = f"Bearer {st.secrets['NVIDIA_API_KEY']}"  # Enter API Key directly here

UPLOAD_ASSET_TIMEOUT = 300  # Timeout for asset upload
MAX_RETRIES = 5  # Max polling retries
DELAY_BTW_RETRIES = 1  # Delay between polls

def _upload_asset(input_data, description):
    assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
    headers = {
        "Authorization": header_auth,
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    s3_headers = {
        "x-amz-meta-nvcf-asset-description": description,
        "content-type": "image/jpeg",
    }
    payload = {"contentType": "image/jpeg", "description": description}

    # Request to upload asset
    response = requests.post(assets_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    asset_url = response.json()["uploadUrl"]
    asset_id = response.json()["assetId"]

    # Upload image to asset URL
    response = requests.put(asset_url, data=input_data, headers=s3_headers, timeout=UPLOAD_ASSET_TIMEOUT)
    response.raise_for_status()

    return uuid.UUID(asset_id)

def capture_image_from_camera():
    """Capture an image from the user's camera and return it as an image object."""
    st.text("Click to take a picture")
    camera_image = st.camera_input("Take a Picture")
    
    if camera_image:
        # Convert camera image to PIL Image and return it
        return Image.open(camera_image)
    return None

# Streamlit page layout setup
st.set_page_config(page_title="Grounding Dino Object Detection", layout="wide")

# Sidebar with navigation tabs
tab = st.sidebar.radio("Navigate", ["Home", "Processing", "History"])

# Initialize session state to store file history
if "history" not in st.session_state:
    st.session_state.history = []

# Home Tab - Description of the model
if tab == "Home":
    st.title("Grounding Dino Object Detection")
    st.write("""
        This model uses NVIDIA's **Grounding Dino** for object detection. You can upload an image, enter a prompt for object detection, 
        and the model will analyze the image based on your input. The system supports real-time image capture via webcam and provides 
        downloadable results with detected objects.
    """)
    st.write("### Features:")
    st.write("- Real-time image capture from the camera")
    st.write("- Object detection using a custom prompt")
    st.write("- Download result images in JPG or PNG format")

# Processing Tab - Image upload, camera capture, and analysis
elif tab == "Processing":
    st.title("Object Detection - Processing")
    
    # Input prompt and file upload in Streamlit
    prompt = st.text_input("Enter the prompt for object detection:")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Option for taking a real-time picture with the camera
    camera_image = capture_image_from_camera()

    if st.button("Analyze Image"):
        image_to_analyze = None

        # Check if camera image or uploaded image is provided
        if uploaded_image:
            image_to_analyze = Image.open(uploaded_image)
        elif camera_image:
            image_to_analyze = camera_image
        elif not prompt:
            st.error("Please enter a prompt and upload or capture an image.")
            st.stop()

        if image_to_analyze and prompt:
            # Convert image to bytes and upload to NVIDIA API
            img_bytes = BytesIO()
            image_to_analyze.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            asset_id = _upload_asset(img_bytes.read(), "Input Image")

            # Prepare the inputs for the object detection model
            inputs = {
                "model": "Grounding-Dino",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "media_url", "media_url": {"url": f"data:image/jpeg;asset_id,{asset_id}"}}
                        ]
                    }
                ],
                "threshold": 0.3
            }

            asset_list = f"{asset_id}"
            headers = {
                "Content-Type": "application/json",
                "NVCF-INPUT-ASSET-REFERENCES": asset_list,
                "NVCF-FUNCTION-ASSET-IDS": asset_list,
                "Authorization": header_auth,
            }

            # Make a request to the NVIDIA API
            response = requests.post(nvai_url, headers=headers, json=inputs)

            if response.status_code == 200:
                # Save the zip output file
                with open(f"{output_dir}/output.zip", "wb") as out:
                    out.write(response.content)
                st.success("Output saved successfully!")

            elif response.status_code == 202:
                st.write("Pending evaluation ...")
                nvcf_reqid = response.headers['NVCF-REQID']
                poll_url = nvai_polling_url + nvcf_reqid

                # Polling to check if response is ready
                retries = MAX_RETRIES
                while retries > 0:
                    headers_polling = {"accept": "application/json", "Authorization": header_auth}
                    response_polling = requests.get(poll_url, headers=headers_polling)
                    
                    if response_polling.status_code == 202:
                        st.write("Result is not yet ready. Polling...")
                        retries -= 1
                        time.sleep(DELAY_BTW_RETRIES)
                    elif response_polling.status_code == 200:
                        # Save the zip output file
                        with open(f"{output_dir}/output.zip", "wb") as out:
                            out.write(response_polling.content)
                        st.success("Result ready!")
                        break
                    else:
                        st.error(f"Unexpected response status: {response_polling.status_code}")
                        break

            # Unzip and display the output files
            with zipfile.ZipFile(f"{output_dir}/output.zip", "r") as z:
                z.extractall(output_dir)

            # List and display extracted files
            extracted_files = os.listdir(output_dir)
            st.write("Extracted Files:", extracted_files)

            # Find the first image file in the output directory
            image_file = next((file for file in extracted_files if file.endswith((".jpg", ".png"))), None)

            if image_file:
                st.image(os.path.join(output_dir, image_file), caption="Detected Objects")
                
                # Option to download the result image in .jpg or .png
                result_image_path = os.path.join(output_dir, image_file)
                result_image = Image.open(result_image_path)

                # Add download button for results in different formats
                download_format = st.radio("Choose download format", ["JPG", "PNG"])
                if download_format == "JPG":
                    img_bytes = BytesIO()
                    result_image.save(img_bytes, format="JPEG")
                    img_bytes.seek(0)
                    st.download_button("Download JPG", data=img_bytes, file_name="result.jpg")
                elif download_format == "PNG":
                    img_bytes = BytesIO()
                    result_image.save(img_bytes, format="PNG")
                    img_bytes.seek(0)
                    st.download_button("Download PNG", data=img_bytes, file_name="result.png")
                
                # Store the result in history
                st.session_state.history.append({"file": image_file, "status": "Done"})
            else:
                st.error("No image found in the extracted output. Please check the output files.")
        else:
            st.error("Please enter a prompt and upload or capture an image.")

# History Tab - Display file analysis history
elif tab == "History":
    st.title("Analysis History")
    
    if st.session_state.history:
        for entry in st.session_state.history:
            status = "✔️" if entry["status"] == "Done" else "❌"
            st.write(f"{entry['file']} - {status}")
    else:
        st.write("No history available. Please analyze an image first.")
