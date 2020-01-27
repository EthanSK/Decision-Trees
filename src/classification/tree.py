from ..util.data_set import Dataset
from ..util.data_read import data_read
from eval import Evaluator
from collections import Counter
import math  # piazza says this is allowed
import numpy as np
from pathlib import Path
import pickle
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

    def predict(self, features) -> str:
        return self.traverse_until_leaf(features=features, node=self.root_node)

    def traverse_until_leaf(self, features, node: NodeBinTree):
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
            node, false_set, true_set = self.find_best_node(
                dataset)  # fix false set true set not worknig
            if node is None:
                return NodeBinTree(NodeData(label=self.find_majority_label(dataset)))
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
        false_set_min, true_set_min = None, None
        for feature_idx in range(num_features):
            sorted_entries = sorted(
                dataset.entries, key=lambda en: en.features[feature_idx])
            # sorted_entries = self.counting_sort(dataset, feature_idx)
            prev_entry = None
            for i in range(len(sorted_entries)):
                entry = sorted_entries[i]
                if prev_entry is not None and entry.label != prev_entry.label:
                    test_node = NodeBinTree(NodeData(
                        lt_operand_feature_idx=feature_idx, gt_operand=entry.features[feature_idx]))
                    false_set, true_set = self.split_dataset(
                        test_node, dataset)
                    child_entropy_combined = len(false_set.entries)/len(dataset.entries) * \
                        self.calc_entropy(false_set.entries) + \
                        len(true_set.entries)/len(dataset.entries) * \
                        self.calc_entropy(true_set.entries)
                    if child_entropy_combined < min_entropy and sorted_entries[0].features[feature_idx] < entry.features[feature_idx]:
                        min_entropy = child_entropy_combined
                        node_min_entropy = test_node
                        false_set_min = false_set
                        true_set_min = true_set
                prev_entry = entry

        if node_min_entropy is not None:
            node_min_entropy.data.set_entropy(min_entropy)
        return node_min_entropy, false_set_min, true_set_min

    def calc_entropy(self, entries):
        label_counts = Counter(
            [entry.label for entry in entries])
        entropy = 0
        for label in label_counts:
            probability = label_counts[label] / len(entries)
            entropy += -probability * math.log2(probability)
        return entropy

# first get all nodes where children are all leaves
# then test each one and remove if accuracy is higher

    def prune(self, node: NodeBinTree, val_feats, val_lbls, ev: Evaluator):
        def accuracy():
            conf = ev.confusion_matrix([self.predict(f)
                                        for f in val_feats], val_lbls)
            return ev.accuracy(conf)

        if node.false_child.data.label is not None and node.true_child.data.label is not None:
            acc_before = accuracy()
            node.data.label = node.false_child.data.label  # no majority since 2 children
            acc_after = accuracy()
            if acc_after <= acc_before:
                node.data.label = None  # back to being a non leaf node
                # print("prune failed")
            else:
                print("pruned leaves", node.false_child.data,
                      node.true_child.data, " into ", node.data)
                print("acc before: ", acc_before, "acc after: ", acc_after)

        if node.false_child.data.label is None:
            f = self.prune(node.false_child, val_feats, val_lbls, ev)
        if node.true_child.data.label is None:
            t = self.prune(node.true_child,  val_feats, val_lbls, ev)

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
