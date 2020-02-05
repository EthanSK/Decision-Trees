from .tree import BinTree
from ..util.data_read import data_read
from pathlib import Path
from turtle import *


def visualize_tree(tree: BinTree, max_depth: int = 10, save_filename: str = None):
    tree_str = tree.__repr__(max_depth=max_depth)
    print(tree_str)
    if save_filename is not None:
        Path("out").mkdir(parents=True, exist_ok=True)
        f = open("out/" + save_filename, "w+")
        f.write(tree_str)
        f.close()

def image_visualize_tree(tree: BinTree, max_depth: int = 10, save_filename: str = None):
    t = Turtle()
    len_ang = 20
    t.penup()
    t.setpos(-300, -200)
    t.pendown()
    recursive_draw(tree.root_node, 1, len_ang, t)


def recursive_draw(node, current_depth, len_ang, t):
    if current_depth >= 10: return
    t.forward((10-current_depth)/2 * len_ang)
    if(node.true_child is not None): 
        t.left((len_ang)/current_depth)
        recursive_draw(node.true_child, current_depth + 1, len_ang,t)
    if(node.false_child is not None): 
        t.right((len_ang)/current_depth)
        recursive_draw(node.false_child, current_depth + 1, len_ang,t)
    t.backward((10-current_depth)/2 * len_ang)
    ts = t.getscreen()
    ts.getcanvas().postscript(file="duck.eps")
    return

if __name__ == "__main__":
    dataset = data_read("data/toy.txt")
    tree = BinTree(dataset)
    visualize_tree(tree, max_depth=3, save_filename="visualize_tree_full.txt")

