from styx_msgs.msg import TrafficLight
import cv2
from keras.models import load_model
from numpy import newaxis
import numpy as np
import tensorflow as tf 
import os

class TLClassifier(object):
    def __init__(self):
        path = os.getcwd()
        self.model = load_model(path+'/light_classification/real_tl_classifier_new.h5') 
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        image = cv2.resize(image, (800, 600)) 
        image = image.astype(float)
        image = image / 255.0
        scores = self.model.predict(image)
        #print('Classification:' ,classification[0])

        if(scores[0] == 1): # 'Traffic Light: Red', 'Traffic Light: Green/Yellow', 'No Traffic Light' or [0,1,2]
            
            print("Classifier predicted RED light")
            return TrafficLight.RED
        
        print("Classifier predicted NOT red light")
        return TrafficLight.UNKNOWN