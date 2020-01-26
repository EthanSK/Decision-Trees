from __future__ import annotations
from ..util.data_set import Dataset
from ..util.data_read import data_read
from collections import Counter
import math  # piazza says this is allowed
import numpy as np
from pathlib import Path
import pickle


class NodeData:
    def __init__(self, label: str = None, lt_operand_feature_idx: int = None, gt_operand: float = None):
        self.label = label
        # we store the operands needed for a < condition check
        self.lt_operand_feature_idx = lt_operand_feature_idx
        self.gt_operand = gt_operand

    def set_entropy(self, value: float):
        self.entropy = value

    def __repr__(self):
        if self.label is not None:
            return f"Leaf: {self.label}"
        else:
            return f"| x_{self.lt_operand_feature_idx} < {self.gt_operand} | Child entropy: {'%.2f' % self.entropy} |"


class NodeBinTree:
    def __init__(self, data: NodeData, false_child: NodeBinTree = None, true_child: NodeBinTree = None):
        self.data = data
        self.false_child = false_child
        self.true_child = true_child

    def set_false_child_node(self, node: NodeBinTree):
        self.false_child = node

    def set_true_child_node(self, node: NodeBinTree):
        self.true_child = node

    def __repr__(self, level=0, max_depth=10):
        indent = "    " * (level + 1)
        is_max_depth_exceeded = level >= max_depth - 1
        max_depth_warning_msg = "✋"
        extra_msg = max_depth_warning_msg if is_max_depth_exceeded else ""
        string = f"[L{level}] {self.data} {extra_msg} \n"
        if is_max_depth_exceeded:
            return string
        if self.true_child is not None:
            string += f"{indent} ✅ {self.true_child.__repr__(level+1, max_depth)}"
        if self.false_child is not None:
            string += f"{indent} ❌ {self.false_child.__repr__(level+1, max_depth)}"
        return string


class BinTree:
    def __init__(self, dataset: Dataset = None, should_load_file: bool = False):
        if should_load_file:
            try:
                self.load_tree()
            except:
                self.root_node = self.induce_decision_tree(dataset)
        else:
            self.root_node = self.induce_decision_tree(dataset)

    def __repr__(self, max_depth: int):
        return self.root_node.__repr__(level=0, max_depth=max_depth)

    def predict(self, features: Array) -> str:
        return self.traverse_until_label(features=features, node=self.root_node)

    def traverse_until_label(self, features: Array, node: NodeBinTree):
        if node.data.label is not None:
            return node.data.label
        if features[node.data.lt_operand_feature_idx] < node.data.gt_operand:
            return self.traverse_until_label(features, node.true_child)
        else:
            return self.traverse_until_label(features, node.false_child)

    def find_majority_label(self, dataset: Dataset):
        entry_labels = [entry.label for entry in dataset.entries]
        max_value = 0
        label_max_value = None
        label_counts = {lb: entry_labels.count(
            lb) for lb in np.unique(entry_labels)}
        for label in label_counts:
            if label_counts[label] > max_value:
                max_value = label_counts[label]
                label_max_value = label
        return label

    def induce_decision_tree(self, dataset: Dataset):
        if all(x.label == dataset.entries[0].label for x in dataset.entries):
            return NodeBinTree(NodeData(label=dataset.entries[0].label))
        else:
            node = self.find_best_node(dataset)
            false_set, true_set = self.split_dataset(node, dataset)

            # check if data can't be split any further
            if np.array_equal(false_set.entries, dataset.entries):
                return NodeBinTree(NodeData(label=self.find_majority_label(dataset)))
            else:
                node.set_false_child_node(
                    self.induce_decision_tree(false_set))

            if np.array_equal(true_set.entries, dataset.entries):
                return NodeBinTree(NodeData(label=self.find_majority_label(dataset)))
            else:
                node.set_true_child_node(self.induce_decision_tree(true_set))

            return node

    def split_dataset(self, node: NodeBinTree, dataset: Dataset) -> Array[Dataset, Dataset]:
        true_set, false_set = [], []
        lt_f_idx = node.data.lt_operand_feature_idx  # the feature to use
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
            feature_col = [entry.features[feature_idx]
                           for entry in dataset.entries]
            sorted_entry_indices = np.argsort(feature_col)
            prev_entry = None
            for entry_idx in sorted_entry_indices:
                entry = dataset.entries[entry_idx]
                if prev_entry is None or entry.label != prev_entry.label:
                    # the feature idx is feature_idx, the operand is entry.features[entry_idx][feature_idx]]
                    # construct a potential 'test' node to calculate entropy against and see if min entropy
                    test_node = NodeBinTree(NodeData(
                        lt_operand_feature_idx=feature_idx, gt_operand=entry.features[feature_idx]))
                    false_set, true_set = self.split_dataset(
                        test_node, dataset)
                    # we don't need to calculate the entropy of the parent node in order to find IG coz it's the same
                    child_entropy_combined = len(false_set.entries)/len(dataset.entries) * \
                        self.calc_entropy(false_set) + \
                        len(true_set.entries)/len(dataset.entries) * \
                        self.calc_entropy(true_set)
                    if child_entropy_combined < min_entropy:
                        min_entropy = child_entropy_combined
                        node_min_entropy = test_node
                prev_entry = entry
        node_min_entropy.data.set_entropy(min_entropy)
        return node_min_entropy

    def calc_entropy(self, dataset: Dataset):
        label_counts = Counter(
            [entry.label for entry in dataset.entries])
        entropy = 0
        for label in label_counts:
            probability = label_counts[label] / len(dataset.entries)
            entropy += -probability * math.log2(probability)
        return entropy

    def prune_tree(self):
        count = 0
        while prev_tree != self.root_node:
            self.root_node = prev_tree
            prune_leaf(self.root_node, count)
            count += 1

    def prune_leaf(self, node: NodeBinTree, count: int):
        pred1, pred2 = False, False
        if node.false_child.label is not None:
            if count == 0:
                return NodeBinTree(NodeData(label=node.data.label))
            else:
                pred1 = True
                count -= 1
        if node.true_child.label is not None:
            if count == 0:
                return NodeBinTree(NodeData(label=node.data.label))
            else:
                pred2 = True
                count -= 1
        if not pred1:
            false_node = self.prune_leaf(node.false_child)
        if not pred2:
            true_node = self.prune_leaf(node.true_child)

        return node

    def save_tree(self, filename: str = "trained_tree.obj"):
        Path("out").mkdir(parents=True, exist_ok=True)
        f = open("out/" + filename, "wb")
        pickle.dump(self.root_node, f)
        f.close()

    def load_tree(self, filename: str = "trained_tree.obj"):
        f = open("out/" + filename, 'rb')
        self.root_node = pickle.load(f)
        f.close()


if __name__ == "__main__":
    dataset = data_read("data/toy.txt")
    tree = BinTree(dataset)
    pass
