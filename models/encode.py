LE_VECT="le_vect.mdl"

import joblib as j
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Encoder():
    le_vect = None
    
    def __init__(self):
        self.le_vect = j.load(LE_VECT)
        
    def transform(self,X):
        return self.encode(X)
        
    def encode(self,arr):
        if type(arr) != list: arr=list(arr)
        #get = set(arr)
        #ref = set(self.le_vect.classes_)
        #get = list(get.intersection(ref))
        
        vect = [self.le_vect.transform(arr)]#.reshape(-1,1)
        vect = pad_sequences(vect,maxlen=130+1)
        return vect.ravel()




    
