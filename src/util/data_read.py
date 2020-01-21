##############################################################################
# Task 1.1 
# Part of answer - Data read method
##############################################################################

import numpy as np
from .data_set import data_set

def data_read(filename: str) -> data_set:
    """Returns two numpy arrays containing features and ground truths
    respectively. Arrays are dynamically sized based on amount of 
    features in the given .txt file.

    Args:
        filname -- a string giving the path to the .txt data file
    Returns:
        data_set -- a data_set object containing both the attributes
            np array and the features np array
    """
    features, ground_truths = [], []
    file_object = open(filename, "r+")
    for line in file_object:
        temp = [attr.strip() for attr in line.split(',')]
        features.append(temp[0:-1])
        ground_truths.append(temp[-1])
    
    return data_set(
              np.asarray(features, dtype=np.int),
              np.asarray(ground_truths, dtype=np.str)
           )