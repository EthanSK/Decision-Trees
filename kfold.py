from src.classification.tree import BinTree
from eval import Evaluator
from src.util.data_read import data_read
from src.util.data_set import Dataset
import numpy as np
import random


def kfold(dataset: Dataset, k):
    ev = Evaluator()
    subsets = dataset.split_k_subsets(k)
    accuracies = []
    unique_lbls = np.unique([e.label for e in dataset.entries])
    for i in range(k):
        validation = subsets[i]
        accuracies_inner = []
        for j in range(k):
            if i == j:
                continue
            tree = BinTree(subsets[j])
            val_feats, val_lbls = validation.shim_to_arrays()
            preds = [tree.predict(f) for f in val_feats]
            conf_mat = ev.confusion_matrix(preds, val_lbls, unique_lbls)
            accuracies_inner.append(ev.accuracy(conf_mat))
        accuracies.append(accuracies_inner)
    return accuracies


def kfold_average_std(accuracies):
    folds_accuracies = [np.mean(fold_res) for fold_res in accuracies]
    average = np.mean(folds_accuracies)
    std = np.std(folds_accuracies)
    return (average, std)


def kfold_best_subset_vs_full():
    ev = Evaluator()
    matrix = ev.confusion_matrix(preds, y_test)
    print("real accuracy: ", accuracy_score(y_test, preds))
    print("\nour calc accuracy: ", str.format('{0:.15f}', ev.accuracy(matrix)))
    print("\n precision:", precision_score(y_test, preds, average="macro"))
    print("\n our precision: ", ev.precision(matrix))
    print("\nreal recall: ", recall_score(y_test, preds, average="macro"))
    print("\n our recall: ", ev.recall(matrix))
    print("\n f1_score", f1_score(y_test, preds, average="macro"))
    print("\n f1_score: ", ev.f1_score(matrix))


if __name__ == "__main__":
    train_file = "train_sub"
    dataset = data_read(f"data/{train_file}.txt")
    accs = kfold(dataset, 10)
    average, std = kfold_average_std(accs)
    print("kfold average: ", average, " ± ", std)
