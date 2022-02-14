import streamlit as st
import PIL
import numpy as np
import pandas as pd
from tensorflow import keras
import cv2
from PIL import Image
import glob


@st.cache
def testing(A, model_name):
    model = keras.models.load_model(model_name)
    A = A.reshape(64,32,1)
    sub, pos = model.predict(np.expand_dims(A,0))
    sub = sub.argmax(-1)
    pos = pos.argmax(-1)
    n_par = model.count_params()
    #visualkeras.layered_view(model, to_file='model.png').show() # write to disk
    return sub, pos, n_par

@st.cache
def read_csv(path):
    info = pd.read_csv(path, names=['Number','Age','Height','Weight'],encoding = "utf16", header=0)
    return info

st.title('HDA Project Live Demo')
img_file_buffer = st.file_uploader('Upload an image', type=['png','jpg','jpeg'])

col1, col2, col3 = st.beta_columns([2,4,2])
if img_file_buffer is not None:
    img_orig = img_file_buffer
    image = PIL.Image.open(img_orig)
    img_array = np.array(image)
    img_resized = cv2.resize(img_array,(32,32),interpolation = cv2.INTER_AREA)
    with col2:
        st.write("Image Uploaded Successfully!", width=10)
        
dict = {1:'Supine',8:'Supine Star',9:'Supine Hand Crossed',10:'Supine Knees Up',
                   11:'Supine Right Knee Up',12:'Supine Left Knee Up',15:'Supine 30° Bed Inclination',16:'Supine 45° Bed Inclination',
                   17:'Supine 60° Bed Inclination',2:'Right',4:'Right 30° Body Roll',5:'Right 60° Body Roll',13:'Right Fetus',
                  3:'Left',6:'Left 30° Body Roll',7:'Left 60° Body Roll',14:'Left Fetus'}

col1, col2, col3 = st.beta_columns([2,2,4])
with col1:
    if img_file_buffer is not None:
        shape = st.radio("Choose image shape",   ('(64,32)','(32,32)'),index=0)

        if shape == '(64,32)':
            st.image(img_array,'Demo Image',width=300)
        else:
            st.image(img_resized,'Demo Image',width=300)

with col3:
    option = st.selectbox('Select Model', ('NN','CNN', 'RNN', 'CNN+RNN', 'LSTM','CNN+LSTM'), index=0)
    if option == 'CNN':
        model_name = 'cnn.h5'
    if option == 'RNN':
        model_name = 'rnn.h5'
    if option == 'NN':
        model_name = 'nn.h5'
    if option == 'CNN+RNN':
        model_name = 'cnn_rnn.h5'
    if option == 'LSTM':
        model_name = 'lstm.h5'
    if option == 'CNN+LSTM':
        model_name = 'cnn_lstm.h5'

    if img_file_buffer is not None:
        sub, pos, n_par = testing(img_array/1000, model_name)
        st.write('')
        st.write('')
        position = dict[int(pos)+1]
        st.write('Position:',f'<p style="color:lightgreen;font-size:15px;border-radius:2%;">{position}</p>', unsafe_allow_html=True)
        st.write('')
        st.write('')
        row = int(sub)
        info = read_csv('info.csv')
        sub_info = info.iloc[row,:]
        st.write('Subject Info:',sub_info)
        st.write('')
        st.write('')
        st.write('N° parameters:', n_par)
        #st.image('model.png')


