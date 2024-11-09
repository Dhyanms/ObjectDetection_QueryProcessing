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
import base64
import json

# NVIDIA API endpoints and authorization
nvai_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nv-grounding-dino"
nvai_polling_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
neva_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
header_auth = f"Bearer {st.secrets['NVIDIA_API_KEY']}"

UPLOAD_ASSET_TIMEOUT = 300
MAX_RETRIES = 5
DELAY_BTW_RETRIES = 1

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

    response = requests.post(assets_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    asset_url = response.json()["uploadUrl"]
    asset_id = response.json()["assetId"]

    response = requests.put(asset_url, data=input_data, headers=s3_headers, timeout=UPLOAD_ASSET_TIMEOUT)
    response.raise_for_status()

    return uuid.UUID(asset_id)

def capture_image_from_camera():
    st.text("Click to take a picture")
    camera_image = st.camera_input("Take a Picture")
    
    if camera_image:
        return Image.open(camera_image)
    return None

def get_image_description(image_b64, query):
    headers = {
        "Authorization": header_auth,
        "Accept": "text/event-stream"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'{query} <img src="data:image/png;base64,{image_b64}" />'
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.70,
        "seed": 0,
        "stream": True
    }

    response = requests.post(neva_url, headers=headers, json=payload)
    
    result = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data: "):
                try:
                    data = json.loads(decoded_line[6:])
                    if "choices" in data:
                        choice = data["choices"][0]
                        if "delta" in choice and "content" in choice["delta"]:
                            result += choice["delta"]["content"]
                except json.JSONDecodeError:
                    pass

    return result

# Streamlit page layout setup
st.set_page_config(page_title="NVIDIA Vision Assistant", layout="wide")

# Sidebar with navigation tabs
tab = st.sidebar.radio("Navigate", ["Home", "Processing", "History"])

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "detected_image" not in st.session_state:
    st.session_state.detected_image = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None

# Home Tab
if tab == "Home":
    st.title("NVIDIA Vision Assistant")
    st.write("""
        This application combines two powerful NVIDIA AI models:
        1. **Grounding Dino** for object detection
        2. **NEVA-22B** for answering questions about detected objects
        
        Upload an image, detect objects, and then ask questions about what you see!
    """)
    st.write("### Features:")
    st.write("- Real-time image capture from camera")
    st.write("- Object detection using custom prompts")
    st.write("- Natural language queries about detected objects")
    st.write("- Download results in JPG or PNG format")

# Processing Tab
elif tab == "Processing":
    st.title("Vision Processing")
    
    # Object Detection Section
    st.header("Step 1: Object Detection")
    prompt = st.text_input("Enter the prompt for object detection:")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    camera_image = capture_image_from_camera()

    if st.button("Detect Objects"):
        image_to_analyze = None

        if uploaded_image:
            image_to_analyze = Image.open(uploaded_image)
            st.session_state.original_image = image_to_analyze
        elif camera_image:
            image_to_analyze = camera_image
            st.session_state.original_image = camera_image
        
        if image_to_analyze and prompt:
            img_bytes = BytesIO()
            image_to_analyze.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            asset_id = _upload_asset(img_bytes.read(), "Input Image")

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

            response = requests.post(nvai_url, headers=headers, json=inputs)

            if response.status_code in [200, 202]:
                if response.status_code == 202:
                    st.write("Processing...")
                    nvcf_reqid = response.headers['NVCF-REQID']
                    poll_url = nvai_polling_url + nvcf_reqid

                    retries = MAX_RETRIES
                    while retries > 0:
                        headers_polling = {"accept": "application/json", "Authorization": header_auth}
                        response_polling = requests.get(poll_url, headers=headers_polling)
                        
                        if response_polling.status_code == 202:
                            st.write("Still processing...")
                            retries -= 1
                            time.sleep(DELAY_BTW_RETRIES)
                        elif response_polling.status_code == 200:
                            response = response_polling
                            break
                        else:
                            st.error(f"Error: {response_polling.status_code}")
                            break

                # Save and extract results
                with open(f"{output_dir}/output.zip", "wb") as out:
                    out.write(response.content)

                with zipfile.ZipFile(f"{output_dir}/output.zip", "r") as z:
                    z.extractall(output_dir)

                # Find and display the output image
                image_file = next((f for f in os.listdir(output_dir) if f.endswith((".jpg", ".png"))), None)
                
                if image_file:
                    result_image_path = os.path.join(output_dir, image_file)
                    st.session_state.detected_image = Image.open(result_image_path)
                    st.image(st.session_state.detected_image, caption="Detected Objects")
                    st.session_state.history.append({"file": image_file, "status": "Done"})
                    
                    # Modified download options with correct format handling
                    download_format = st.radio("Choose download format", ["JPEG", "PNG"])
                    img_bytes = BytesIO()
                    st.session_state.detected_image.save(img_bytes, format=download_format)
                    img_bytes.seek(0)
                    
                    # Use .jpg extension for JPEG format
                    file_extension = "jpg" if download_format == "JPEG" else "png"
                    st.download_button(
                        f"Download {download_format}", 
                        data=img_bytes, 
                        file_name=f"result.{file_extension}"
                    )
    # Query Section
    if st.session_state.detected_image:
        st.header("Step 2: Ask Questions")
        user_query = st.text_input("Enter your question about the image:")
        
        if st.button("Get Answer"):
            if user_query:
                # Convert image to base64
                img_bytes = BytesIO()
                st.session_state.original_image.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                image_b64 = base64.b64encode(img_bytes.getvalue()).decode()

                # Get and display the answer
                with st.spinner("Processing your question..."):
                    result = get_image_description(image_b64, user_query)
                    st.subheader("Answer:")
                    st.write(result)
            else:
                st.warning("Please enter a question about the image.")

# History Tab
elif tab == "History":
    st.title("Analysis History")
    
    if st.session_state.history:
        for entry in st.session_state.history:
            status = "✔️" if entry["status"] == "Done" else "❌"
            st.write(f"{entry['file']} - {status}")
    else:
        st.write("No history available. Please analyze an image first.")