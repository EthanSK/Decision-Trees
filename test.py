from src.util.data_read import data_read 
from src.evaluation.eval import Evaluator

if __name__ == "__main__":
    #data_read("data/toy.txt")
    prediction = ["A","B","B","C","C","B"]
    annotation = ["A","D","B","C","A","B"]
    class_labels = ["B","A","C","D","E"]
    obj = Evaluator()
    matrix = obj.confusion_matrix(prediction, annotation, class_labels)
    print(obj.accuracy(matrix))
    print(obj.precision(matrix))
    print(obj.recall(matrix))
    print(obj.f1_score(matrix))