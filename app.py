import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import ManTraNet

st.title("🔍 ManTra-Net Fake Image Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image.resize((512, 512))) / 255.0

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Running ManTra-Net...")
    inp = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()

    net = ManTraNet()
    net.load_state_dict(torch.load("models/ManipulationTracing_pretrained.pth", map_location="cpu"))
    net.eval()

    with torch.no_grad():
        out = net(inp)[0, 0].numpy()

    fig, ax = plt.subplots()
    heatmap = ax.imshow(out, cmap='jet')
    fig.colorbar(heatmap)
    st.pyplot(fig)
