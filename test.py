from src.util.data_read import data_read 
from src.evaluation.eval import Evaluator

if __name__ == "__main__":
    #data_read("data/toy.txt")
    prediction = ["A","B"]
    annotation = ["A","A"]
    class_labels = ["B","A"]
    obj = Evaluator()
    matrix = obj.confusion_matrix(prediction, annotation, class_labels)
    print(obj.accuracy(matrix))
    print(obj.precision(matrix))
    print(obj.recall(matrix))
    print(obj.f1_score(matrix))