##############################################################################
# Task 1.1
# Part of answer - Dataset class
##############################################################################

import numpy as np


class DataEntry:
    def __init__(self, features, label: str):
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

    def __init__(self, data_entries):
        self.entries = data_entries

    def __repr__(self):
        return str(self.entries)

    def shim_to_arrays(self):
        return (np.array([entry.features for entry in self.entries]), np.array([entry.label for entry in self.entries]))

    def split_k_subsets(self, k):
        np.random.shuffle(self.entries)
        return [Dataset(entries) for entries in np.array_split(self.entries, k)]
