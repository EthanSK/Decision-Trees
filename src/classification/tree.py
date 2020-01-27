from ..util.data_set import Dataset
from ..util.data_read import data_read
from collections import Counter
import math  # piazza says this is allowed
import numpy as np
from pathlib import Path
import pickle
from nptyping import Array
import time


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
            return f"| x_{self.lt_operand_feature_idx} < {self.gt_operand} | ChldEntr: {'%.2f' % self.entropy} |"


class NodeBinTree:
    def __init__(self, data: NodeData, false_child=None, true_child=None):
        self.data = data
        self.false_child = false_child
        self.true_child = true_child

    def set_false_child_node(self, node):
        self.false_child = node

    def set_true_child_node(self, node):
        self.true_child = node

    def __repr__(self, level=0, max_depth=10):
        indent = "    " * (level + 1)
        is_max_depth_exceeded = level >= max_depth - 1
        max_depth_warning_msg = "!MDE"  # max depth exceeded
        extra_msg = max_depth_warning_msg if is_max_depth_exceeded else ""
        string = f"[L{level}] {self.data} {extra_msg} \n"
        if is_max_depth_exceeded:
            return string
        if self.true_child is not None:
            string += f"{indent} T: {self.true_child.__repr__(level+1, max_depth)}"
        if self.false_child is not None:
            string += f"{indent} F: {self.false_child.__repr__(level+1, max_depth)}"
        return string


class BinTree:
    def __init__(self, dataset: Dataset = None, saved_tree_file: str = None):
        if saved_tree_file is not None:
            try:
                self.load_tree(filename=saved_tree_file)
            except:
                self.root_node = self.induce_decision_tree(dataset)
        else:
            self.root_node = self.induce_decision_tree(dataset)

    def __repr__(self, max_depth: int):
        return self.root_node.__repr__(level=0, max_depth=max_depth)

    def predict(self, features: Array) -> str:
        return self.traverse_until_leaf(features=features, node=self.root_node)

    def traverse_until_leaf(self, features: Array, node: NodeBinTree):
        if node.data.label is not None:
            return node.data.label
        if features[node.data.lt_operand_feature_idx] < node.data.gt_operand:
            return self.traverse_until_leaf(features, node.true_child)
        else:
            return self.traverse_until_leaf(features, node.false_child)

    def find_majority_label(self, dataset: Dataset):
        max_value = 0
        label_max_value = None
        label_counts = Counter(
            [entry.label for entry in dataset.entries])
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
            if node is None:
                return NodeBinTree(NodeData(label=self.find_majority_label(dataset)))
            false_set, true_set = self.split_dataset(node, dataset)
            node.set_false_child_node(
                self.induce_decision_tree(false_set))
            node.set_true_child_node(self.induce_decision_tree(true_set))
            return node

    def split_dataset(self, node: NodeBinTree, dataset: Dataset) -> [Dataset, Dataset]:
        true_set, false_set = [], []
        lt_f_idx = node.data.lt_operand_feature_idx  # the feature to use
        gt_op = node.data.gt_operand
        for entry in dataset.entries:
            true_set.append(
                entry) if entry.features[lt_f_idx] < gt_op else false_set.append(entry)
        return [Dataset(false_set), Dataset(true_set)]

    def find_best_node(self, dataset: Dataset) -> NodeBinTree:
        start_time = time.time()
        num_features = len(dataset.entries[0].features)
        min_entropy = math.inf
        node_min_entropy = None
        for feature_idx in range(num_features):
            feature_col = [entry.features[feature_idx]
                           for entry in dataset.entries]
            # print("feature_col", len(feature_col))
            sorted_entry_indices = np.argsort(feature_col)
            prev_entry = None
            for entry_idx in sorted_entry_indices:
                entry = dataset.entries[entry_idx]
                if prev_entry is not None and entry.label != prev_entry.label:
                    # the feature idx is feature_idx, the operand is entry.features[entry_idx][feature_idx]]
                    # construct a potential 'test' node to calculate entropy against and see if min entropy
                    test_node = NodeBinTree(NodeData(
                        lt_operand_feature_idx=feature_idx, gt_operand=entry.features[feature_idx]))
                    # false_set, true_set = self.split_dataset(
                    #     test_node, dataset)  # why do we need to call this here? it does the same thing as the current loop. we can be efficient. also calc_entrpy loops through all entries. there must be a way to combine functionality to just this loop
                    # we don't need to calculate the entropy of the parent node in order to find IG coz it's the same
                    # child_entropy_combined = len(false_set.entries)/len(dataset.entries) * \
                    #     self.calc_entropy(false_set) + \
                    #     len(true_set.entries)/len(dataset.entries) * \
                    #     self.calc_entropy(true_set)
                    child_entropy_combined = self.calc_entropy_children(
                        dataset, test_node)
                    if child_entropy_combined < min_entropy and dataset.entries[sorted_entry_indices[0]].features[feature_idx] != entry.features[feature_idx]:
                        min_entropy = child_entropy_combined
                        node_min_entropy = test_node
                prev_entry = entry
        if node_min_entropy is not None:
            node_min_entropy.data.set_entropy(min_entropy)
        # print("durationnnn: ", time.time() - start_time)
        return node_min_entropy

    def calc_entropy(self, dataset: Dataset):
        label_counts = Counter(
            [entry.label for entry in dataset.entries])
        entropy = 0
        for label in label_counts:
            probability = label_counts[label] / len(dataset.entries)
            entropy += -probability * math.log2(probability)
        return entropy

    def calc_entropy_children(self, dataset: Dataset, test_node: NodeBinTree):
        # label_counts = Counter(
        # [entry.label for entry in dataset.entries])
        # we need a separate label count for each set.
        # entropy = 0
        # for label in label_counts:
        #     probability = label_counts[label] / len(dataset.entries)
        #     entropy += -probability * math.log2(probability)

        false_label_counts, true_label_counts = {}, {}
        lt_f_idx = test_node.data.lt_operand_feature_idx  # the feature to use
        gt_op = test_node.data.gt_operand
        false_set_entropy, true_set_entropy = 0, 0
        false_set_size, true_set_size = 0, 0
        for entry in dataset.entries:
            if entry.features[lt_f_idx] < gt_op:
                true_set_size += 1
                true_label_counts[entry.label] = true_label_counts.get(
                    entry.label, 0) + 1
            else:
                false_set_size += 1
                false_label_counts[entry.label] = false_label_counts.get(
                    entry.label, 0) + 1

        for label in false_label_counts:
            probability = false_label_counts[label] / false_set_size
            false_set_entropy += -probability * math.log2(probability)

        for label in true_label_counts:
            probability = true_label_counts[label] / true_set_size
            true_set_entropy += -probability * math.log2(probability)

        # for entry in dataset.entries:
        #     if entry.features[lt_f_idx] < gt_op:
        #         probability = true_label_counts[entry.label] / true_set_size
        #         true_set_entropy += -probability * math.log2(probability)
        #     else:
        #         probability = false_label_counts[entry.label] / false_set_size
        #         false_set_entropy += -probability * math.log2(probability)
        return (false_set_size / len(dataset.entries)) * false_set_entropy + (true_set_size / len(dataset.entries)) * true_set_entropy

    def prune_tree(self):
        count = 0
        while prev_tree != self.root_node:
            self.root_node = prev_tree
            prune_leaf(self.root_node, count)
            count += 1

    # def prune(self, features: Array, node: NodeBinTree):
    #     if node.data.label is not None:
    #         return node.data.label
    #     if features[node.data.lt_operand_feature_idx] < node.data.gt_operand:
    #         return self.traverse_until_leaf(features, node.true_child)
    #     else:
    #         return self.traverse_until_leaf(features, node.false_child)

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
    dataset = data_read("data/train_sub.txt")
    tree = BinTree(dataset)
    pass
