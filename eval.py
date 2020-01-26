##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks:
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np
from nptyping import Array
import math


class Evaluator(object):
    """ Class to perform evaluation
    """

    def confusion_matrix(self, prediction: Array, annotation: Array,
                         class_labels: Array = None) -> Array:
        """ Computes the confusion matrix.

        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.

        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        if not class_labels:
            class_labels = np.unique(annotation)

        confusion = np.zeros(
            (len(class_labels), len(class_labels)), dtype=np.float)

        #######################################################################
        #                 ** TASK 3.1: COMPLETE THIS METHOD **
        #######################################################################

        unique = ((class_labels))
        imap = {key: i for i, key in enumerate(unique)}

        for p, a in zip(prediction, annotation):
            confusion[imap[p]][imap[a]] += 1

        return confusion

    def accuracy(self, confusion: Array) -> float:
        """ Computes the accuracy given a confusion matrix.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """

        #######################################################################
        #                 ** TASK 3.2: COMPLETE THIS METHOD **
        #######################################################################

        # true_pos = np.diag(confusion)
        # false_pos = np.sum(confusion, axis=0) - true_pos
        # false_neg = np.sum(confusion, axis=1) - true_pos
        # true_neg = np.sum(confusion) - (true_pos + false_pos + false_neg)

        # accuracy_array = (true_neg + true_pos) / \
        #     (true_neg + true_pos + false_neg + false_pos)

        # # divide macro accuracy by number of classes to normalise
        # accuracy = np.sum(accuracy_array) / len(confusion)

        accurate = confusion.trace()
        accuracy = float(str(float(accurate/np.sum(confusion)))[0:5])

        return accuracy

    def precision(self, confusion: Array) -> (Array, float):
        """ Computes the precision score per class given a confusion matrix.

        Also returns the macro-averaged precision across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        p : np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        macro_p : float
            The macro-averaged precision score across C classes.
        """

        # Initialise array to store precision for C classes
        # NOTA BENE: We do not use this
        p = np.zeros((len(confusion), ))

        #######################################################################
        #                 ** TASK 3.3: COMPLETE THIS METHOD **
        #######################################################################

        true_pos = np.diag(confusion)
        false_pos = np.sum(confusion, axis=0) - true_pos

        # where= attribute specifies not to divide by zero
        p = np.divide(true_pos, (false_pos + true_pos),
                      where=(false_pos+true_pos != 0))
        macro_p = p.sum() / len(confusion)

        return (p, macro_p)

    def recall(self, confusion: Array) -> (Array, float):
        """ Computes the recall score per class given a confusion matrix.

        Also returns the macro-averaged recall across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged recall score across C classes.
        """

        # Initialise array to store recall for C classes
        # NOTA BENE: we don't use this
        r = np.zeros((len(confusion), ))

        #######################################################################
        #                 ** TASK 3.4: COMPLETE THIS METHOD **
        #######################################################################

        true_pos = np.diag(confusion)
        false_neg = np.sum(confusion, axis=1) - true_pos

        # where= attribute specifies not to divide by zero
        r = np.divide(true_pos, (true_pos + false_neg),
                      where=(true_pos+false_neg) != 0)
        macro_r = r.sum() / len(confusion)

        return (r, macro_r)

    def f1_score(self, confusion: Array) -> (Array, float):
        """ Computes the f1 score per class given a confusion matrix.

        Also returns the macro-averaged f1-score across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged f1 score across C classes.
        """

        # Initialise array to store recall for C classes
        # NOTA BENE: we don't use this
        f = np.zeros((len(confusion), ))

        #######################################################################
        #                 ** YOUR TASK: COMPLETE THIS METHOD **
        #######################################################################

        true_pos = np.diag(confusion)
        false_pos = np.sum(confusion, axis=0) - true_pos
        false_neg = np.sum(confusion, axis=1) - true_pos

        p = np.divide(true_pos, (false_pos + true_pos),
                      where=(false_pos+true_pos != 0))
        macro_p = p.sum() / len(confusion)

        r = np.divide(true_pos, (true_pos + false_neg),
                      where=(true_pos+false_neg) != 0)
        macro_r = r.sum() / len(confusion)

        f = 2 * np.divide(p*r, p+r, where=(p+r) != 0)

        # This definiton is valid but not the one used in the coursework as per the specification
        # macro_f = (2 * macro_p * macro_r) / (macro_p + macro_r)

        macro_f = f.sum() / len(confusion)

        return (f, macro_f)
