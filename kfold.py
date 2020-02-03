from src.classification.tree import BinTree
from eval import Evaluator
from src.util.data_read import data_read
from src.util.data_set import Dataset
import numpy as np
import random
from collections import Counter


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


def kfold_best_subset_vs_full(accuracies, trees, train_file: str):
    ev = Evaluator()

    # it's fine if the prediction accuracy of max_sub_tree is significatntly lower than the average accuracy of the kfold results, because we are testing against test data here, in kfold the test data was part of the actual set, so it makes sense if it got higher prediction accuracy.
    max_sub_tree = trees[np.argmax(accuracies)]

    dataset = data_read(f"data/{train_file}.txt")
    full_tree = BinTree(dataset, f"tree_{train_file}.obj")

    test_dataset = data_read("data/test.txt")
    unique_lbls = np.unique([e.label for e in test_dataset.entries])
    x_test, y_test = test_dataset.shim_to_arrays()
    sub_tree_preds = [max_sub_tree.predict(f) for f in x_test]
    full_tree_preds = [full_tree.predict(f) for f in x_test]

    sub_tree_matrix = ev.confusion_matrix(sub_tree_preds, y_test, unique_lbls)
    full_tree_matrix = ev.confusion_matrix(
        full_tree_preds, y_test, unique_lbls)

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


def combined_pred(trees, train_file: str):
    ev = Evaluator()

    test_dataset = data_read("data/test.txt")
    unique_lbls = np.unique([e.label for e in test_dataset.entries])
    x_test, y_test = test_dataset.shim_to_arrays()

    combined_preds = []
    for row in x_test:
        preds = []
        print(len(combined_preds))
        for tree in trees:
            preds.append(tree.predict(row))
        vote = Counter(preds).most_common(1)[0][0]
        combined_preds.append(vote)
    
    dataset = data_read(f"data/{train_file}.txt")
    full_tree = BinTree(dataset, f"tree_{train_file}.obj")
    full_tree_preds = [full_tree.predict(f) for f in x_test]
    assert len(combined_preds) == len(full_tree_preds)

    combined_tree_matrix = ev.confusion_matrix(combined_preds, y_test, unique_lbls)
    full_tree_matrix = ev.confusion_matrix(full_tree_preds, y_test, unique_lbls)

    print("\nsub_tree_preds our calc accuracy: ",
          str.format('{0:.15f}', ev.accuracy(combined_tree_matrix)))
    print("\combined_tree_preds our precision: ", ev.precision(combined_tree_matrix))
    print("\combined_tree_preds our recall: ", ev.recall(combined_tree_matrix))
    print("\combined_tree_preds f1_score: ", ev.f1_score(combined_tree_matrix))

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
    # kfold_best_subset_vs_full(accs, trees, train_file)
    combined_pred(trees, train_file)