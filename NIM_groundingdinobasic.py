import time
import uuid
import zipfile
import requests
import streamlit as st
import os

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

st.title("Grounding Dino Object Detection")

# Input prompt and file upload in Streamlit
prompt = st.text_input("Enter the prompt for object detection:")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

if st.button("Analyze Image"):
    if uploaded_image and prompt:
        # Upload image to NVIDIA API
        asset_id = _upload_asset(uploaded_image.read(), "Input Image")

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
            # Add download button for results
            st.download_button("Download Results", data=open(f"{output_dir}/output.zip", "rb"), file_name="results.zip")
        else:
            st.error("No image found in the extracted output. Please check the output files.")
    else:
        st.error("Please enter a prompt and upload an image.")
