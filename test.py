from src.util.data_read import data_read
from src.evaluation.eval import Evaluator
from src.classification.visualize_tree import visualize_tree
from src.classification.tree import BinTree
from classification import DecisionTreeClassifier
import sys
import time


def test_DecisionTreeClassifier():
    # train
    start = time.time()
    cl = DecisionTreeClassifier(should_load_file=False)
    dataset = data_read("data/train_sub.txt")
    x, y = dataset.shim_to_arrays()
    cl.train(x, y)
    cl.tree.save_tree()
    visualize_tree(cl.tree)
    duration = time.time() - start
    print("duration: ", duration)

    # predict
    val_dataset = data_read("data/validation.txt")
    x_val, y_val = val_dataset.shim_to_arrays()
    preds = cl.predict(x_val)

    # evaluate
    ev = Evaluator()
    matrix = ev.confusion_matrix(preds, y_val)
    print("\naccuracy: ", ev.accuracy(matrix))
    print("\n precision: ", ev.precision(matrix))
    print("\n recall: ", ev.recall(matrix))
    print("\n f1_score: ", ev.f1_score(matrix))


def test_tree_load_file():
    tree = BinTree(should_load_file=True)
    visualize_tree(tree)


def run_manual_test():
    start = time.time()
    dataset = data_read("data/simple1.txt")
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
    test_DecisionTreeClassifier()
    # test_tree_load_file()
    # run_manual_test()
