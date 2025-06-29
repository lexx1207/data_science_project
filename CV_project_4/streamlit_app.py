# streamlit run main.py
import streamlit as st
import os
from PIL import Image
from streamlit_image_select import image_select
from utils.adain import adain
from utils.capvstn import capvstn
from models.RevResNet import RevResNet
from models.adainnet import Model, VGGEncoder, RC, Decoder, denorm

st.title('Style Transfer')
content_folder = '/mount/src/data_science_project/CV_project_4/example_image/content/'
style_folder = '/mount/src/data_science_project/CV_project_4/example_image/style/'
output_image = '/mount/src/data_science_project/CV_project_4/out_content/output.png'
content_files = []
style_files = []
for file in os.listdir(content_folder):
    if file.endswith(".jpg"):
        content_files.append(os.path.join(content_folder, file))
for file in os.listdir(style_folder):
    if file.endswith(".jpg"):
        style_files.append(os.path.join(style_folder, file))


img1 = image_select("Select image", content_files)
st.write(img1)

img2 = image_select("Select image style", style_files)


model_name = st.sidebar.selectbox(
    'Select Model',
    ('ADAIN', 'CAP-VSTN')
)


clicked = st.button('Stylize')

if clicked:
    if model_name == 'ADAIN':
        adain(img1,img2)
    if model_name == 'CAP-VSTN':
        capvstn(img1,img2)
    
    st.write('### Output image:')
    image = Image.open(output_image)
    st.image(image, width=400)

