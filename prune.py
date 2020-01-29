from src.classification.tree import BinTree
from eval import Evaluator
from src.util.data_read import data_read
import random


def prune(tree: BinTree):
    vld_dataset = data_read("data/validation.txt")
    x_val, y_val = vld_dataset.shim_to_arrays()
    ev = Evaluator()
    for i in range(10):
        print(f"----prune attempt {i + 1}---")
        tree.prune(node=tree.root_node, og_vld_feats=x_val,
                   og_vld_lbls=y_val, dataset=vld_dataset, ev=ev)


if __name__ == "__main__":
    train_file = "train_noisy"
    dataset = data_read(f"data/{train_file}.txt")
    tree = BinTree(dataset, f"tree_{train_file}.obj")
    test_dataset = data_read("data/test.txt")

    ev = Evaluator()
    x_test, y_test = test_dataset.shim_to_arrays()
    preds = [tree.predict(x) for x in x_test]
    matrix = ev.confusion_matrix(preds, y_test)
    print("test accuracy before pruning:",  ev.accuracy(matrix))

    prune(tree)
    preds = [tree.predict(x) for x in x_test]
    matrix = ev.confusion_matrix(preds, y_test)
    print("test accuracy after pruning:",  ev.accuracy(matrix))
