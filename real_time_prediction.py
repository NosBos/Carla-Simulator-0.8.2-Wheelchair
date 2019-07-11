import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import glob
import numpy as np
from keras.models import model_from_json


class RealTimePrediction: 
      
    def __init__(self, model_name, model_weights):
        self.model_name = model_name
        self.model_weights = model_weights
        self.predicted_value = None
        self.model = self.load_model()
        self.current_image = None
        self._AI_frame = 0
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
        self._AI_frame += 1
        resize_img = cv2.resize(img,(160,320))
        reshaped_img = resize_img.reshape(1,160,320,3)
        
        #reshape makes unsavable image, saving the img after resizing, before reshaping
        ai_filename = 'Auto/episode{}.jpg'.format(self._AI_frame)
        
        #saves resized img to disk
        cv2.imwrite(ai_filename,resize_img):
        
        return reshaped_img
       
    # Calling the model to do prediction
    def do_predict(self, img):   
        reshape_img = self.read_data(img)

        self.current_image = reshape_img        

        self.predicted_value = self.model.predict(reshape_img)
        return self.predicted_value 




