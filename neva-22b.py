import streamlit as st
import requests
import base64
import json

# Define the API endpoint and model
invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
stream = True  # Set to True to use streaming response

# Function to handle image and query processing
def get_image_description(image_b64, query):
    # Set the headers for the request with the API key
    headers = {
        "Authorization": f"Bearer {st.secrets['NVIDIA_API_KEY']}",  # Enter API Key directly here,  # Replace with your actual API key
        "Accept": "text/event-stream" if stream else "application/json"
    }

    # Prepare the payload with the base64-encoded image and the user query
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
        "stream": stream
    }

    # Send the POST request to the API
    response = requests.post(invoke_url, headers=headers, json=payload)

    result = ""
    if stream:
        # Handle streaming response
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                # Look for data and parse it into a usable format
                if decoded_line.startswith("data: "):
                    # Extract content after "data: "
                    try:
                        data = json.loads(decoded_line[6:])
                        if "choices" in data:
                            choice = data["choices"][0]
                            if "delta" in choice and "content" in choice["delta"]:
                                result += choice["delta"]["content"]
                    except json.JSONDecodeError:
                        pass  # Handle any errors in decoding JSON data
    else:
        # If no stream, just return the JSON response
        result = response.json()

    return result

# Streamlit UI
st.title("Image Description with NEVA-22B Model")

# Image upload section
uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# Text input for user query
user_query = st.text_input("Enter a query for the model:")

# Process the image and query when both are provided
if uploaded_image and user_query:
    # Read and encode the image to base64
    image = uploaded_image.read()
    image_b64 = base64.b64encode(image).decode()

    # Call the model to get a response
    with st.spinner("Processing your query..."):
        result = get_image_description(image_b64, user_query)

    # Show the result from the model
    st.subheader("Model Response:")
    st.write(result)
