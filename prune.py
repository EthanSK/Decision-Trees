from src.classification.tree import BinTree
from eval import Evaluator
from src.util.data_read import data_read


def prune(tree: BinTree):
    val_dataset = data_read("data/validation.txt")
    x_val, y_val = val_dataset.shim_to_arrays()
    ev = Evaluator()
    for i in range(10):
        print("-------")
        tree.prune(node=tree.root_node, val_feats=x_val, val_lbls=y_val, ev=ev)


if __name__ == "__main__":

    train_file = "train_full"
    dataset = data_read(f"data/{train_file}.txt")
    tree = BinTree(train_file, f"tree_{train_file}.obj")
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
