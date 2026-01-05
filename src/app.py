import streamlit as st
import torch
import os
from generator import Generator

# -------- PATH SETUP --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "..", "checkpoints")

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(
        torch.load(os.path.join(CHECKPOINTS_DIR, "G_final.pth"))
    )
    model.eval()
    return model

G = load_model()

# -------- UI --------
st.title("Synthetic Image Generator (Vanilla GAN)")
st.write("Privacy-Preserving Synthetic Image Generation")

num_images = st.slider("Number of images to generate", 1, 10, 5)

if st.button("Generate Images"):
    z = torch.randn(num_images, 100)
    images = G(z)

    for img in images:
        st.image(img[0].detach().numpy(), clamp=True)
