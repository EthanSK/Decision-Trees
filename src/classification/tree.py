from __future__ import annotations

from src.util.data_set import DataSet


class NodeBinTree:
    def __init__(self, data, left_node: NodeBinTree = None, right_node: NodeBinTree = None):
        self.data = data  # dunno what this is yet
        self.left_node = left_node
        self.right_node = right_node


class BinTree:
    def __init__(self):
        self.root = NodeBinTree()

    def induce(self, dataset: DataSet):
        pass
