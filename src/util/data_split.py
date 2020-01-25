##############################################################################
# Task 3.6
# Full answer
##############################################################################

import numpy as np
from nptyping import Array
from typing import List

# Can't type hint List of numpy.array


def data_split(array: Array, k: int):
    """Returns a list of k numpy arrays, which are randomly split
    subsections of array.

    Args:
        array -- numpy.array to be split
        k -- number of sub-arrays to make from main array
    Returns:
        sub_arraays -- a list of k numpy.arrays which form a
            partition of array
    """

    if k > len(array) or k == 1:
        return array
    # as np.random.choice doesn't allow k = population size
    elif k == len(array):
        sub_arrays = []
        for el in array:
            sub_arrays.append(np.array(el))
        return sub_arrays

    split_points = np.random.choice(len(array) - 2, k - 1, replace=False) + 1
    split_points.sort()
    sub_arrays = np.split(array, split_points)

    return sub_arrays


if __name__ == "__main__":
    # test case
    data_split(np.array([1, 1, 1, 4, 6, 9, 1, 1, 3, 4, 6, 1]), 12)
