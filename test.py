from src.util.data_read import data_read
from src.evaluation.eval import Evaluator
from src.classification.visualize_tree import visualize_tree
from src.classification.tree import BinTree
import time

if __name__ == "__main__":
    start = time.time()
    dataset = data_read("data/toy.txt")
    tree = BinTree(dataset)
    visualize_tree(tree, max_depth=50, save_filename="tree.txt")
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
