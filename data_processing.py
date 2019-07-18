
import sys
import csv
import cv2

class DataProcessing:
    
    def read_csv_data(csv_file):
        file_lines = []
        with open(csv_file + "") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                file_lines.append(line)       
        return file_lines        
    
    def get_image_path(img_string, image_dir):
        img_path = image_dir + img_string.split('/')[-1] + '.jpg'
        return img_path
    
    def read_image(img_path):
        img = cv2.imread(img_path)
        return img
    
    def display_img(img):
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def crop_img(img):
        if img is not None: 
            y, x, d = img.shape
            croped_img = img[int(y/2):, :]       
        return croped_img
    
    def preprocess_image(img):
        if img is not None:             
            croped_img = DataProcessing.crop_img(img) 
            resized_img = cv2.resize(croped_img, (200, 100)) 
        return resized_img 
    
    def augment_data():
        return   



