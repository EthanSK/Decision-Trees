from __future__ import annotations
from ..util.data_set import Dataset
from ..util.data_read import data_read
import collections  # piazza says this is allowed
import math  # piazza says this is allowed
import numpy as np


class NodeData:
    def __init__(self, label: str = None, lt_operand_feature_idx: int = None, gt_operand: float = None):
        self.label = label
        # we store the operands needed for a < condition check
        self.lt_operand_feature_idx = lt_operand_feature_idx
        self.gt_operand = gt_operand

    def __repr__(self):
        if self.label is not None:
            return f"Leaf: {self.label}"
        else:
            return f"x_{self.lt_operand_feature_idx} < {self.gt_operand}"


class NodeBinTree:
    def __init__(self, data: NodeData, false_child: NodeBinTree = None, true_child: NodeBinTree = None):
        self.data = data
        self.false_child = false_child
        self.true_child = true_child

    def set_false_child_node(self, node: NodeBinTree):
        self.false_child = node

    def set_true_child_node(self, node: NodeBinTree):
        self.true_child = node

    def __repr__(self, level, max_depth):
        indent = "    " * (level + 1)
        is_max_depth_exceeded = level >= max_depth - 1
        max_depth_warning_msg = "✋"
        extra_msg = max_depth_warning_msg if is_max_depth_exceeded else ""
        string = f"{self.data} [L{level}] {extra_msg} \n"
        if is_max_depth_exceeded:
            return string
        if self.false_child is not None:
            string += f"{indent} ❌ {self.false_child.__repr__(level+1, max_depth)}"
        if self.true_child is not None:
            string += f"{indent} ✅ {self.true_child.__repr__(level+1, max_depth)}"
        return string


class BinTree:
    def __init__(self, dataset: Dataset):
        self.root_node = self.induce_decision_tree(dataset)

    def __repr__(self, max_depth: int):
        return self.root_node.__repr__(level=0, max_depth=max_depth)

    def induce_decision_tree(self, dataset: Dataset):
        if len(dataset.entries) == 1 or all(x.label == dataset.entries[0].label for x in dataset.entries):
            return NodeBinTree(NodeData(label=dataset.entries[0].label))
        else:
            node = self.find_best_node(dataset)
            false_child, true_child = self.split_dataset(node, dataset)
            node.set_false_child_node(self.induce_decision_tree(false_child))
            node.set_true_child_node(self.induce_decision_tree(true_child))
            return node

    def split_dataset(self, node: NodeBinTree, dataset: Dataset) -> Array[Dataset, Dataset]:
        true_set, false_set = [], []
        lt_f_idx = node.data.lt_operand_feature_idx  # which feature to use
        gt_op = node.data.gt_operand
        for entry in dataset.entries:
            true_set.append(
                entry) if entry.features[lt_f_idx] < gt_op else false_set.append(entry)
        return [Dataset(false_set), Dataset(true_set)]

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
                    test_node = NodeBinTree(NodeData(
                        lt_operand_feature_idx=feature_idx, gt_operand=entry.features[feature_idx]))
                    false_child, true_child = self.split_dataset(
                        test_node, dataset)
                    test_node.set_false_child_node(false_child)
                    test_node.set_true_child_node(true_child)
                    # we don't need to calculate the entropy of the parent node in order to find IG coz it's the same
                    child_entropy_combined = \
                        len(false_child.entries)/len(dataset.entries) * \
                        self.calc_entropy(false_child) + \
                        len(true_child.entries)/len(dataset.entries) * \
                        self.calc_entropy(true_child)
                    if child_entropy_combined < min_entropy:
                        min_entropy = child_entropy_combined
                        node_min_entropy = test_node
                prev_entry = entry
        return node_min_entropy

    def calc_entropy(self, dataset: Dataset):
        label_counts = collections.Counter(
            [entry.label for entry in dataset.entries])
        entropy = 0
        for label in label_counts:
            probability = label_counts[label] / len(dataset.entries)
            entropy += -probability * math.log2(probability)
        return entropy


if __name__ == "__main__":
    dataset = data_read("data/toy.txt")
    tree = BinTree(dataset)
    pass
