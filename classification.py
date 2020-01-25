##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the
# DecisionTreeClassifier
##############################################################################
from __future__ import annotations

import numpy as np
from nptyping import Array
from src.util.data_set import Dataset, DataEntry
from src.classification.tree import BinTree


class DecisionTreeClassifier(object):
    """
    A decision tree classifier

    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained

    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X

    """

    def __init__(self, should_load_file=False):
        self.is_trained = False
        self.should_load_file = should_load_file

    def train(self, x: Array, y: Array) -> DecisionTreeClassifier:
        """ Constructs a decision tree classifier from data

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array of labels

        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance

        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################
        data_entries = []
        for i in range(len(x)):
            data_entries.append(DataEntry(features=x[i], label=y[i]))
        self.dataset = Dataset(data_entries)

        self.tree = BinTree(
            self.dataset, should_load_file=self.should_load_file)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        return self

    def predict(self, x: Array) -> Array:
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the
            number of attributes)

        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception(
                "Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)

        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        predictions = [self.tree.predict(features) for features in x]
        # remember to change this if you rename the variable
        return predictions
