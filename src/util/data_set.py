##############################################################################
# Task 1.1 
# Part of answer - Dataset class
##############################################################################

import numpy as np
from nptyping import Array

class data_set():
    """
    Holds both attributes and golden standard scraped by 
    dataRead from .txt files  

    Attributes
    ----------
    features : Array
        NumPy array holding the 16 features for each entry
    ground_truths : Array
        NumPy array holding the ground truth values for each entry    
    """
    def __init__(self, features: Array, ground_truths: Array):
        self.features = features
        self.ground_truths = ground_truths