from __future__ import annotations
from ..util.data_set import Dataset
from ..util.data_read import data_read
import collections  # piazza says this is allowed
import math  # piazza says this is allowed
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
    def __init__(self):
        # self.root = NodeBinTree()
        pass

    def induce_decision_tree(self, dataset: Dataset):
        if len(dataset.entries) == 1 or all(x.label == dataset.entries[0].label for x in dataset.entries):
            return NodeBinTree(DataNode(label=dataset.entries[0].label))
        else:
            node = self.find_best_node(dataset)
            false_child, true_child = self.split_dataset(node, dataset)
            node.set_false_child_node(induce_decision_tree(false_child))
            node.set_false_child_node(induce_decision_tree(true_child))
            return node

    def split_dataset(self, node: NodeBinTree, dataset: Dataset) -> Array[NodeBinTree, NodeBinTree]:
        true_set, false_set = [], []
        lt_f_idx = node.data.lt_operand_feature_idx  # which feature to use
        gt_op = node.data.gt_operand
        for entry in dataset.entries:
            true_set.append(
                entry) if entry.features[lt_f_idx] < gt_op else false_set.append(entry)
        return [false_set, true_set]

    def find_best_node(self, dataset: Dataset) -> NodeBinTree:
        num_features = len(dataset.entries[0].features)
        min_entropy = math.inf
        node_min_entropy = None
        for feature_idx in range(num_features):
            sorted_entry_indices = np.argsort(
                [entry.features[feature_idx] for entry in dataset.entries])
            prev_entry = None
            for entry_idx in sorted_entry_indices:
                entry = dataset.entries[entry_idx]
                if prev_entry is None or entry.label != prev_entry.label:
                    # the feature idx is feature_idx, the operand is entry.features[entry_idx][feature_idx]]
                    # construct a potential 'test' node to calculate entropy against and see if min entropy
                    test_node = NodeBinTree(DataNode(
                        lt_operand_feature_idx=feature_idx, gt_operand=entry.features[feature_idx]))
                    false_child, true_child = self.split_dataset(
                        test_node, dataset)
                    test_node.set_false_child_node(false_child)
                    test_node.set_true_child_node(true_child)
                    # we don't need to calculate the entropy of the current node in order to find IG coz it's the same
                    child_entropy_combined = \
                        len(false_child.entries)/len(dataset.entries) * \
                        self.calc_entropy(false_child) + \
                        len(true_child.entries)/len(dataset.entries) * \
                        self.calc_entropy(true_child)
                    if child_entropy_combined < min_entropy:
                        min_entropy = child_entropy_combined
                        node_min_entropy = test_node
                prev_entry = entry

    def calc_entropy(self, dataset: Dataset):
        label_counts = collections.Counter(
            [entry.label for entry in dataset.entries])
        entropy = 0
        for label, count in label_counts:
            probability = count / len(dataset.entries)
            entropy += -probability * math.log2(probability)
        return entropy

    def calc_info_gain(self, node: NodeBinTree):
        # does node entrop - (entrop of children ie average entrop of childern)


if __name__ == "__main__":
    dataset = data_read("data/toy.txt")
    tree = BinTree()
    tree.find_best_node(dataset)
