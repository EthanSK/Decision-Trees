from src.classification.tree import BinTree
from eval import Evaluator
from src.util.data_read import data_read


def prune(tree: BinTree):
    val_dataset = data_read("data/validation.txt")
    x_val, y_val = val_dataset.shim_to_arrays()
    ev = Evaluator()
    for i in range(10):
        tree.prune(node=tree.root_node, val_feats=x_val, val_lbls=y_val, ev=ev)
        print("\n\n-------")


if __name__ == "__main__":
    train_file = "train_full"
    dataset = data_read(f"data/{train_file}.txt")
    tree = BinTree(train_file, f"tree_{train_file}.obj")
    prune(tree)
