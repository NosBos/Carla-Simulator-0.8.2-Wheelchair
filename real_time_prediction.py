import sys
import glob
import numpy as np
from data_processing import DataProcessing
from keras.models import model_from_json


class RealTimePrediction: 
      
    def __init__(self, model_arc, model_weights):
		#model arc is json file
        self.model_arc = model_arc
        self.model_weights = model_weights
        self.predicted_value = None
        self.model = self.load_model()

         
    # Load json file, create the model and load the weights
    def load_model(self):  
        json_file = open(self.model_arc, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json) # Load the model    
        model.load_weights(self.model_weights) # Load weights into new model
        print("Loaded model from disk")
        return model
       
    # Calling the model to do prediction
    def do_predict(self, img):   
        image = DataProcessing.preprocess_image(img)
        y, x, c = image.shape
        reshaped_img = image.reshape(1, y, x, c)
        self.predicted_value = self.model.predict(reshaped_img)
        return self.predicted_value 




