from .tree import BinTree
from ..util.data_read import data_read
from pathlib import Path

# todo - write to file functionality


def visualize_tree(tree: BinTree, max_deth: int = 10, save_filename: str = None):
    tree_str = str(tree)
    print(tree_str)
    if save_filename is not None:
        Path("out").mkdir(parents=True, exist_ok=True)
        f = open("out/" + save_filename, "w+")
        f.write(tree_str)
        f.close()


if __name__ == "__main__":
    dataset = data_read("data/toy.txt")
    tree = BinTree(dataset)
    visualize_tree(tree, save_filename="tree.txt")
