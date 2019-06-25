import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import glob
import numpy as np
from keras.models import model_from_json


class RealTimePrediction: 
      
    def __init__(self, model_name, model_weights):
        self.model_name = '/home/gill/Carla/CARLA_0.8.2/PythonClient/Model/model.json'
        self.model_weights = '/home/gill/Carla/CARLA_0.8.2/PythonClient/Model/model_weights.h5'
        self.predicted_value = None
        self.model = self.load_model()
        print(type(self.model))
         
    # Load json file, create the model and load the weights
    def load_model(self):  
        json_file = open(self.model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json) # Load the model    
        model.load_weights(self.model_weights) # Load weights into new model
        print("Loaded model from disk")
        return model
    
    # Load data which we want to do prediction on it
    def read_data(self, img):
        reshaped_img = cv2.resize(img,(160,320)).reshape(1,160,320,3)
        return reshaped_img
       
    # Calling the model to do prediction
    def do_predict(self, img):   
        reshape_img = self.read_data(img)
        self.predicted_value = self.model.predict(reshape_img)
        return self.predicted_value 




