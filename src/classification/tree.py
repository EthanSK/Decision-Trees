from __future__ import annotations
from ..util.data_set import Dataset


class NodeBinTree:
    def __init__(self, data, left_node: NodeBinTree = None, right_node: NodeBinTree = None):
        self.data = data  # dunno what this is yet
        self.left_node = left_node
        self.right_node = right_node


class BinTree:
    def __init__(self, dataset):
        self.root = NodeBinTree()
        self.dataset = dataset

    def induce(self):
        if dataset:
            pass
