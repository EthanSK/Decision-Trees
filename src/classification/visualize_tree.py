from .tree import BinTree
from ..util.data_read import data_read

# todo - write to file functionality


def visualize_tree(tree: BinTree, max_deth: int = 10):
    print(tree)


if __name__ == "__main__":
    dataset = data_read("data/train_sub.txt")
    tree = BinTree(dataset)
    visualize_tree(tree)
