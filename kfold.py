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


def process_kfold_results(accuracies):
    folds_accuracies = [np.mean(fold_res) for fold_res in accuracies]
    print("folds_accuracies", folds_accuracies)
    average = np.mean(folds_accuracies)
    std = np.std(folds_accuracies)
    return (average, std)


if __name__ == "__main__":
    train_file = "train_sub"
    dataset = data_read(f"data/{train_file}.txt")
    accs = kfold(dataset, 10)
    average, std = process_kfold_results(accs)
    print("kfold average: ", average, " ± ", std)
