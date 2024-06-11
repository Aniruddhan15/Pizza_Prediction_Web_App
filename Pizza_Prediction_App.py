# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:14:12 2024

@author: aniru
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# loading the saved model
loaded_model = pickle.load(open("C:/DeploymentWebAPP/trained_model.sav",'rb'))


# creating a function for Prediction

def Pizza_prediction(input_data):
    imput_data = np.asarray(input_data)
    input_reshaped = imput_data.reshape(1,-1)
    std_data = sc.transform(input_reshaped)
    prediction = loaded_model.predict(std_data)
    print(prediction)

    if(prediction[0]==0):
        print("The pizza is chicken")
    elif(prediction[0]==1):
        print("The pizza is classic")
    elif(prediction[0]==3):
        print("The pizza is veggie")
    else:
        print("The pizza is hot")
  
def main():
    
    
    # giving a title
    st.title('Pizza Type Prediction Web App')
    
    
    # getting the input data from the user
    
    
    name = st.text_input('Name of the pizza')
    size = st.text_input('size ')
    price = st.text_input('price of the pizza')
    month = st.text_input('month ordered')
    day = st.text_input('day orderd')
    hour = st.text_input('hour ordered')
    minutes = st.text_input('minutes ordered')
    
    
    # code for Prediction
    Pizza_Type = ''
    
    # creating a button for Prediction
    
    if st.button('Pizza Type Result'):
        Pizza_Type = Pizza_prediction([name, size, price, month, day, hour, minutes])
        
        
    st.success(Pizza_Type)
    
if __name__ == '__main__':
    main()
    
    