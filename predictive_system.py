# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# loading the saved model
loaded_model = pickle.load(open("C:/DeploymentWebAPP/trained_model.sav", 'rb'))

input = (1,1,16.00,1,1,11,40)
imput_data = np.asarray(input)
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