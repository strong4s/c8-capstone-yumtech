import numpy as np

class Ravel:
    def __init__(self,X=None,y=None):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        return X.toarray()#.ravel()#.reshape(-1,1)
    
    def fit_transform(self,X,y=None):
        return X.toarray()#.ravel()#.reshape(-1,1)