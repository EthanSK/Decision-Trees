from src.util.data_read import data_read
from eval import Evaluator
from src.classification.visualize_tree import visualize_tree
from src.classification.tree import BinTree
from classification import DecisionTreeClassifier
import sys
import time
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


def run_all():
    files = ["simple1.txt", "simple2.txt",
             "test.txt", "train_sub.txt", "toy.txt"]
    for file in files:
        test_DecisionTreeClassifier(file)


def test_DecisionTreeClassifier(dataset_filename: str = "toy.txt", should_load_file=False):
    # train
    start = time.time()
    cl = DecisionTreeClassifier(should_load_file=should_load_file)
    dataset = data_read("data/" + dataset_filename)
    x, y = dataset.shim_to_arrays()
    cl.train(x, y)
    cl.tree.save_tree()
    visualize_tree(
        cl.tree, save_filename=f"visualize_tree_{dataset_filename[:-4]}.txt", max_depth=30)
    duration = time.time() - start
    print("duration: ", duration)

    # predict
    val_dataset = data_read("data/validation.txt")
    x_val, y_val = val_dataset.shim_to_arrays()
    preds = cl.predict(x_val)
    # preds = [random.choice('ACEGOQ')
    #  for _ in range(len(y_val))]  # testing random
    # evaluate
    ev = Evaluator()
    matrix = ev.confusion_matrix(preds, y_val)
    print("real accuracy: ", accuracy_score(y_val, preds))
    print("\nour calc accuracy: ", str.format('{0:.15f}', ev.accuracy(matrix)))
    print("\n precision:", precision_score(y_val, preds, average="macro"))
    print("\n our precision: ", ev.precision(matrix))
    print("\nreal recall: ", recall_score(y_val, preds, average="macro"))
    print("\n our recall: ", ev.recall(matrix))
    print("\n f1_score", f1_score(y_val, preds, average="macro"))
    print("\n f1_score: ", ev.f1_score(matrix))


def test_tree_load_file():
    tree = BinTree(should_load_file=True)
    visualize_tree(tree)


def run_manual_test():
    start = time.time()
    dataset = data_read("data/train.txt")
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
    print(str.format('{0:.15f}', obj.accuracy(matrix)))
    print(obj.precision(matrix))
    print(obj.recall(matrix))
    print(obj.f1_score(matrix))


if __name__ == "__main__":
    test_DecisionTreeClassifier(
        dataset_filename="simple1.txt", should_load_file=False)
    # test_tree_load_file()
    # run_manual_test()
    # run_all()
