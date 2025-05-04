import streamlit as st
import os
import urllib.request
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from model import ManTraNet

# --- Step 1: Download model from Google Drive if not present ---
model_url = 'https://drive.google.com/uc?export=download&id=1D1v5K4nHl7BqDrCLLFFQf_Pc1C6_zvP4'
model_path = 'ManipulationTracing_pretrained.pth'

if not os.path.exists(model_path):
    with st.spinner("Downloading model weights..."):
        urllib.request.urlretrieve(model_url, model_path)
        st.success("Model downloaded!")

# --- Step 2: Load model ---
@st.cache_resource
def load_model():
    model = ManTraNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

net = load_model()

# --- Step 3: Preprocess image ---
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (400, 400))  # resize to fit model input
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = torch.from_numpy(img).unsqueeze(0)
    return img

# --- Step 4: Run model and show result ---
def predict(img_tensor):
    with torch.no_grad():
        pred = net(img_tensor)
        pred = F.interpolate(pred, size=(400, 400), mode='bilinear', align_corners=False)
        pred = pred.squeeze().cpu().numpy()
        heatmap = (pred * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap_color

# --- Step 5: Streamlit UI ---
st.title("ManTraNet – Image Forgery Detection")

uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img_tensor = preprocess_image(uploaded_file)
    if img_tensor is None:
        st.error("Error decoding image.")
    else:
        st.write("Running forgery detection...")
        result = predict(img_tensor)
        st.image(result, caption="Forgery Heatmap", channels="BGR", use_column_width=True)
