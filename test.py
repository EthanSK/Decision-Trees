from src.util.data_read import data_read
from src.evaluation.eval import Evaluator
from src.classification.visualize_tree import visualize_tree
from src.classification.tree import BinTree
from classification import DecisionTreeClassifier


import time


def test_DecisionTreeClassifier():
    start = time.time()
    cl = DecisionTreeClassifier()
    dataset = data_read("data/test.txt")
    x, y = dataset.shim_to_arrays()
    cl.train(x, y)
    cl.tree.save_tree()
    visualize_tree(cl.tree)
    duration = time.time() - start
    print("duration: ", duration)


def test_tree_load_file():
    tree = BinTree(should_load_file=True)
    visualize_tree(tree)


def run_manual_test():
    start = time.time()
    dataset = data_read("data/toy.txt")
    tree = BinTree(dataset)
    visualize_tree(tree, max_depth=50)
    duration = time.time() - start
    print("duration: ", duration)


def old_test():
    # data_read("data/toy.txt")
    prediction = ["A", "B"]
    annotation = ["A", "A"]
    class_labels = ["B", "A"]
    obj = Evaluator()
    matrix = obj.confusion_matrix(prediction, annotation, class_labels)
    print(obj.accuracy(matrix))
    print(obj.precision(matrix))
    print(obj.recall(matrix))
    print(obj.f1_score(matrix))


if __name__ == "__main__":
    # test_DecisionTreeClassifier()
    test_tree_load_file()
