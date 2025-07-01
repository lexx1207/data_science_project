# streamlit run streamlit_app.py
import streamlit as st
import os
from PIL import Image
from src.open_pose import open_pose
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints

st.title('Virtual Coach')

path_coach = '/mount/src/data_science_project/diplom_2_1/data_input/coach/'
path_user = '/mount/src/data_science_project/diplom_2_1/data_input/user/'
filename_output ='/mount/src/data_science_project/diplom_2_1/data_output/output.png'

ex_number = st.sidebar.selectbox(
    'Select exercise',
    ('Exercise 1', 'Exercise 2','Exercise 3')
)
ex = ex_number.split(" ")[1]

clicked = st.button('Analize')

if clicked:
    open_pose(ex, path_coach, path_user, filename_output)
    
    
    st.write('### Result:')
    image = Image.open(filename_output)
    st.image(image, width=720)