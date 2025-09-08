import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/generate"

st.set_page_config(page_title="Gemini Relay Frontend", page_icon="🤖")

st.title("🤖 Gemini Multimodal Generator")
st.write("Enter text and/or upload an image to get a response.")

# Input prompt
prompt = st.text_area("Your prompt:", height=150)

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if st.button("Generate"):
    if not prompt.strip() and not uploaded_file:
        st.warning("⚠️ Please enter a prompt or upload an image.")
    else:
        with st.spinner("Generating response..."):
            try:
                files = {}
                data = {}

                if prompt.strip():
                    data = {"prompt": prompt}

                if uploaded_file:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

                response = requests.post(API_URL, data=data, files=files)
                if response.status_code == 200:
                    output = response.json().get("output", "")
                    st.success("✅ Response received!")
                    st.text_area("AI Response:", output, height=200)
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
