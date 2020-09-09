import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os, urllib
from PIL import Image
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.models import load_model
import random
from keras.preprocessing import image, sequence
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True



embedding_size=128
max_len=40
vocab_size=8254

@st.cache
def resnet():
    res = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    return res

@st.cache
def im_model():
    image_model = Sequential()
    image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
    image_model.add(RepeatVector(max_len))
    language_model = Sequential()
    language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
    language_model.add(LSTM(256, return_sequences=True))
    language_model.add(TimeDistributed(Dense(embedding_size)))
    conca = Concatenate()([image_model.output, language_model.output])
    x = LSTM(128, return_sequences=True)(conca)
    x = LSTM(512, return_sequences=False)(x)
    x = Dense(vocab_size)(x)
    out = Activation('softmax')(x)
    model = Model([image_model.input, language_model.input],out)
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    model.load_weights('final-imcap.h5')
    return model

page_bg_img = '''
<style>
body {
    background-image: url("https://ak.picdn.net/shutterstock/videos/1016880070/thumb/1.jpg");
    background-size: cover;
    }
    </style>
    '''
st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache
def preprocessing(img_path):
    im = image.load_img(img_path, target_size=(224,224,3))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    return im

def get_encoding(model, img):
    image = preprocessing(img)
    pred = model.predict(image).reshape(2048)
    return pred

tit='''
    <div style="color:black;
              background-color:white;
              font-size:200%;
              font-weight: bold;
              font-style: italic;
              display:inline-block;
              padding:5px;
              border-radius: 15px"
       >Image Captioning</div>
 '''
t=st.markdown(tit, unsafe_allow_html=True)



with open('w2i (2).p', 'rb') as f:
    word_2_indices= pickle.load(f, encoding="bytes")
with open('i2w (2).p', 'rb') as f:
    indices_2_word= pickle.load(f, encoding="bytes")

@st.cache
def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word_2_indices[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        model=im_model()
        preds = model.predict([np.array([image]), np.array(par_caps)])
        word_pred = indices_2_word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])

uploaded_file=st.file_uploader('Upload the image',type=['jpg','png'])
st.set_option('deprecation.showfileUploaderEncoding', False)


if uploaded_file is not None:
    img   = Image.open(uploaded_file)
    st.image(img)
    s=st.success('Generating Caption')
    test_img = get_encoding(resnet(), uploaded_file)
    Argmax_Search = predict_captions(test_img)
    s.empty()
    st.markdown(
        f'''<html>
    <p style="color:white;
              background-color:black;
              font-size:140%;
              display:inline-block;
              padding:10px;
              border-radius: 15px;"
       >{Argmax_Search}</p>
  </html> ''',
        unsafe_allow_html=True)













