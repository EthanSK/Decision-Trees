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
        return f"\nLabel: {self.label} Features: {self.features}"


class Dataset:
    """
    Holds both attributes and golden standard scraped by 
    dataRead from .txt files  
    """

    def __init__(self, data_entries: Array[DataEntry]):
        self.data_entries = data_entries

    def __repr__(self):
        return str(self.data_entries) + "\n"
