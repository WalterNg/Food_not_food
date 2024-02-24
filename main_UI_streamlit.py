import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import time

# Title
st.title("Welcome to Food-not-food classifier!")

# Define Classes
classes = {0:'Food',
           1:"Not Food"}

start = time.time()

model = tf.keras.models.load_model('model\\food_not_food_model_B0-V2.h5')

uploaded_file = st.file_uploader("Choose an image file")


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(uploaded_file)

    image = cv2.resize(image,(224,224))
    # Convert image to tensor
    image = np.expand_dims(image,axis=0)

   # Prediction
    output = model.predict(image)
    pred = classes[round(output[0,0],0)]

    # Define probability
    prob = None
    if pred == "Food":
        prob = round((1-output[0,0])*100,2)
    elif pred =="Not Food":
        prob = round(output[0,0]*100,2)
        
    end = time.time()
    total_time = end - start

    st.write(f"Executed in {round(total_time,2)}s")
    st.markdown(f"{pred}")  
    st.markdown(f"Probability: {prob}%")
    



    


 



