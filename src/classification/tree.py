from __future__ import annotations
from ..util.data_set import Dataset
from ..util.data_read import data_read

import numpy as np


class DataNode:
    def __init__(self, label: str = None, lt_operand_feature_idx: int = None, gt_operand: float = None):
        self.label = label
        # we store the operands needed for a < condition check
        self.lt_operand_feature_idx = lt_operand_feature_idx
        self.gt_operand = gt_operand


class NodeBinTree:
    def __init__(self, data: DataNode, false_child: NodeBinTree = None, true_child: NodeBinTree = None):
        self.data = data
        self.false_child = false_child
        self.true_child = true_child

    def set_false_child_node(self, node: NodeBinTree):
        self.false_child = node

    def set_true_child_node(self, node: NodeBinTree):
        self.true_child = node


class BinTree:
    def __init__(self, dataset: Dataset):
        # self.root = NodeBinTree()
        self.dataset = dataset

    def induce_decision_tree(self):
        if len(dataset.entries) == 1 or all(x.label == dataset.entries[0].label for x in dataset.entries):
            return NodeBinTree(DataNode(label=dataset.entries[0].label))
        else:
            node = self.find_best_node()
            false_child, true_child = self.split_dataset(node)
            node.set_false_child_node(false_child)
            node.set_false_child_node(true_child)
            return node

    def split_dataset(self, node: NodeBinTree):
        true_set, false_set = [], []
        lt_f_idx = node.data.lt_operand_feature_idx  # which feature to use
        gt_op = node.data.gt_operand
        for entry in self.dataset.entries:
            true_set.append(
                x) if entry.features[lt_f_idx] < gt_op else false_set.append(x)
        return [false_set, true_set]

    def find_best_node(self) -> NodeBinTree:
        # we need to iterate over many possible values for the condition, and at each stage calculate the IG. then we return the node with the highest IG.
        # find ig for each row in each column
        # first sort values of the attribute

        # then consider only points that are between two examples in sorted order that have different class labels, while keeping track of the running totals of positive and negative examples on each side of the split point.

        # this is so that instead of testing every single number in existence as a possible split point, we whittle it down to only a few numbers that we need to test. These numbers are the numbers in between

        # we apply a condition. The operands of the condition change every iteration
        # we want to find the operands that give us the highest value for IG (IG is calculated using the 2 new potential subsets)
        # In order to test diferent conditions, we can brute force, or we can be more efficient.

        num_features = len(self.dataset.entries[0].features)
        for i in range(num_features):
            sorted_indices = np.argsort(
                [entry.features[i] for entry in self.dataset.entries])
            sorted_entries = self.dataset.entries[sorted_indices]
            print("i: ", i, sorted_entries)


if __name__ == "__main__":
    tree = BinTree(data_read("data/toy.txt"))
    tree.find_best_node()
