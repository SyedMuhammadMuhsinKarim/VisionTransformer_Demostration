import streamlit as st
from PIL import Image
import torch
from models.yolo import YOLOv5

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

st.title("Tiny Object Detection - Buildings")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    results = model(image)
    results.show()  
    
    st.image(results.render(), caption="Detected Buildings", use_column_width=True)
