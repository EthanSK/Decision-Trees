##############################################################################
# Task 1.1
# Part of answer - Data read method
##############################################################################

import numpy as np
from data_set import DataSet, DataEntry


def data_read(filename: str) -> DataSet:
    """Returns two numpy arrays containing features and ground truths
    respectively. Arrays are dynamically sized based on amount of 
    features in the given .txt file.

    Args:
        filname -- a string giving the path to the .txt data file
    Returns:
        data_set -- a data_set object containing both the attributes
            np array and the features np array
    """
    data_entries = []
    file_object = open(filename, "r+")
    for line in file_object:
        temp = [attr.strip() for attr in line.split(',')]
        data_entries.append(
            DataEntry(features=temp[0:-1], label=temp[-1])
        )

    return DataSet(data_entries)


if __name__ == "__main__":
    # test data_read
    data = data_read("data/toy.txt")
    print(data)
