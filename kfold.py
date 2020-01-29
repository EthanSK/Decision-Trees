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
    trees = []
    unique_lbls = np.unique([e.label for e in dataset.entries])
    for i in range(k):
        test = subsets[i]
        train = Dataset(np.ravel(
            [subsets[j].entries for j in range(len(subsets)) if i != j]))
        tree = BinTree(train)
        test_feats, test_lbls = test.shim_to_arrays()
        preds = [tree.predict(f) for f in test_feats]
        conf_mat = ev.confusion_matrix(preds, test_lbls, unique_lbls)
        accuracies.append(ev.accuracy(conf_mat))
        trees.append(tree)
    return accuracies, trees


def kfold_average_std(accuracies):
    average = np.mean(accuracies)
    std = np.std(accuracies)
    return (average, std)


def kfold_best_subset_vs_full(accuracies, trees):
    ev = Evaluator()
    max_sub_tree = trees[np.argmax(accuracies)]

    train_file = "train_full"
    dataset = data_read(f"data/{train_file}.txt")
    full_tree = BinTree(dataset, f"tree_{train_file}.obj")

    test_dataset = data_read("data/test.txt")
    x_test, y_test = test_dataset.shim_to_arrays()
    sub_tree_preds = [max_sub_tree.predict(f) for f in x_test]
    full_tree_preds = [full_tree.predict(f) for f in x_test]

    sub_tree_matrix = ev.confusion_matrix(sub_tree_preds, y_test)
    full_tree_matrix = ev.confusion_matrix(full_tree_preds, y_test)

    print("\nsub_tree_preds our calc accuracy: ",
          str.format('{0:.15f}', ev.accuracy(sub_tree_matrix)))
    print("\nsub_tree_preds our precision: ", ev.precision(sub_tree_matrix))
    print("\nsub_tree_preds our recall: ", ev.recall(sub_tree_matrix))
    print("\nsub_tree_preds f1_score: ", ev.f1_score(sub_tree_matrix))

    print("\n full_tree_preds our calc accuracy: ",
          str.format('{0:.15f}', ev.accuracy(full_tree_matrix)))
    print("\n full_tree_preds our precision: ", ev.precision(full_tree_matrix))
    print("\n full_tree_preds our recall: ", ev.recall(full_tree_matrix))
    print("\n full_tree_preds f1_score: ", ev.f1_score(full_tree_matrix))


if __name__ == "__main__":
    train_file = "train_full"
    dataset = data_read(f"data/{train_file}.txt")
    accs, trees = kfold(dataset, 10)
    average, std = kfold_average_std(accs)
    print("kfold average: ", average, " ± ", std)
    kfold_best_subset_vs_full(accs, trees)full_treefull_treefull_tree
