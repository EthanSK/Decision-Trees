from .tree import BinTree
from ..util.data_read import data_read

# todo - write to file functionality


def visualize_tree(tree: BinTree, max_deth: int = 10, save_file: str = None):
    print(tree)
    if save_file is not None:
        pass


if __name__ == "__main__":
    dataset = data_read("data/toy.txt")
    tree = BinTree(dataset)
    visualize_tree(tree)
