from .tree import BinTree
from ..util.data_read import data_read
from pathlib import Path

# todo - write to file functionality


def visualize_tree(tree: BinTree, max_depth: int = 10, save_filename: str = None):
    tree_str = tree.__repr__(max_depth=max_depth)
    print(tree_str)
    if save_filename is not None:
        Path("out").mkdir(parents=True, exist_ok=True)
        f = open("out/" + save_filename, "w+")
        f.write(tree_str)
        f.close()


if __name__ == "__main__":
    dataset = data_read("data/toy.txt")
    tree = BinTree(dataset)
    visualize_tree(tree, max_depth=3, save_filename="visualize_tree.txt")
