##############################################################################
# Task 1.1
# Part of answer - Dataset class
##############################################################################

import numpy as np
from nptyping import Array


class DataEntry:
    def __init__(self, features: Array, label: str):
        self.features = features
        self.label = label

    def __repr__(self):  # for printing
        return f"Label: {self.label} Features: {self.features}\n"

    def __eq__(self, other):
        return np.array_equal(self.features, other.features) and self.label == other.label


class Dataset:
    """
    Holds both attributes and golden standard scraped by 
    dataRead from .txt files  
    """

    def __init__(self, data_entries: Array[DataEntry]):
        self.entries = np.array(data_entries)

    def __repr__(self):
        return str(self.entries)

    def shim_to_arrays(self):
        return (np.array([entry.features for entry in self.entries]), np.array([entry.label for entry in self.entries]))
