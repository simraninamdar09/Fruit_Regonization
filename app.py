import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("training_model.h5")
    image = tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])# convert single img to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select page",["Home","About","Fruti Recognition"])

#Home Page
if(app_mode == "Home"):
    st.header("Fruti Detection")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
                

                """)


#About Page
elif(app_mode == "About"):
    st.header("About")
    st.markdown("""
                

                """)
                
    
#Prediction Page
elif(app_mode == "Fruti Recognition"):
    st.header("FRuti" Recognition")
    test_image = st.file_uploader("Choose as Image:")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
    
    #Predict Button
    if(st.button("Predict")):
        with st.spinner("Please Wait..."):
            st.show()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        class_name = ['Apple Braeburn',
              'Apple Granny Smith',
              'Apricot',
              'Avocado',
              'Banana',
              'Blueberry', 
              'Cactus fruit', 
              'Cantaloupe', 
              'Cherry',
              'Clementine',
              'Corn',
              'Cucumber Ripe',
              'Grape Blue',
              'Kiwi',
              'Lemon',
              'Limes',
              'Mango',
              'Onion White',
              'Orange',
              'Papaya',
              'Passion Fruit',
              'Peach',
              'Pear',
              'Pepper Green',
              'Pepper Red',
             'Pineapple',
             'Plum','Pomegranate','Potato Red','Raspberry'
 'Strawberry','Tomato','Watermelon']
        st.success("Model is Predicting It's a {}".format(class_name(result_index)))





