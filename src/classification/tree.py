from __future__ import annotations
from ..util.data_set import Dataset


class DataNode:
    def __init__(self, label: str = None, lt_operand_feature_idx: int, gt_operand: float):
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
        self.root = NodeBinTree()
        self.dataset = dataset

    def induce_decision_tree(self, dataset: Dataset):
        if len(dataset.entries) == 1 or self, x.label == dataset.entries[0].label for x in dataset.entries:
            return NodeBinTree(DataNode(label=dataset.entries[0].label))
        else:
            node = self.find_best_node()
            false_child, true_child = self.split_dataset(node, dataset)
            node.set_false_child_node(false_child)
            node.set_false_child_node(true_child)
            return node

    def split_dataset(self, node: NodeBinTree, dataset: Dataset):
        true_set, false_set = [], []
        lt_f_idx = node.data.lt_operand_feature_idx  # which feature to use
        gt_op = node.data.gt_operand
        for entry in dataset.entries:
            true_set.append(
                x) if entry.features[lt_f_idx] < gt_op else false_set.append(x)
        return [false_set, true_set]

    def find_best_node(self) -> NodeBinTree:
        pass
